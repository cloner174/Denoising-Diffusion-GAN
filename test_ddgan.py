import argparse
import torch
import numpy as np
import os
import torchvision
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from pytorch_fid.fid_score import calculate_fid_given_paths


# Diffusion coefficients functions
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1.0 - torch.exp(2.0 * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input_tensor, t, shape):
    out = torch.gather(input_tensor, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def get_time_schedule(args, device):
    
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.linspace(0, 1, n_timestep + 1, dtype=np.float64)
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small
    
    return t.to(device)



def get_sigma_schedule(args, device):
    
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
    
    t = np.linspace(0, 1, n_timestep + 1, dtype=np.float64)
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8, device=device)
    betas = torch.cat((first[None], betas.to(device)))
    betas = betas.type(torch.float32)
    sigmas = betas.sqrt()
    a_s = (1.0 - betas).sqrt()
    
    return sigmas, a_s, betas



# Posterior coefficients class
class Posterior_Coefficients:
    
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas[1:]
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas
            * self.alphas_cumprod_prev.sqrt()
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * self.alphas.sqrt()
            / (1.0 - self.alphas_cumprod)
        )



def sample_posterior(coefficients, x_0, x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * (0.5 * log_var).exp() * noise
    
    return p_sample(x_0, x_t, t)


def sample_from_model(coefficients, generator, n_time, x_init, T, args):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            latent_z = torch.randn(x.size(0), args.nz, device=x.device)
            x_0 = generator(x, t, latent_z)
            x = sample_posterior(coefficients, x_0, x, t).detach()
    
    return x


def sample_and_test(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #  load saved configuration from training
    exp_path = f'./saved_info/dd_gan/{args.dataset}/{args.exp}'
    content_path = os.path.join(exp_path, 'content.pth')
    
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"Checkpoint {content_path} not found.")
    
    checkpoint = torch.load(content_path, map_location=device)
    saved_args = checkpoint['args']
    # Update args with saved_args
    saved_args.__dict__.update(vars(args))  # Override with any command-line arguments
    args = saved_args
    
    # Function to normalize images to [0, 1]
    to_range_0_1 = lambda x: (x + 1.0) / 2.0
    netG = NCSNpp(args).to(device)
    
    ckpt_path = os.path.join(exp_path, f'netG_{args.epoch_id}.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # remove 'module.' prefix
    if any(key.startswith('module.') for key in ckpt.keys()):
        ckpt = {key.replace('module.', ''): value for key, value in ckpt.items()}
    
    netG.load_state_dict(ckpt, strict=True)
    netG.eval()
    
    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    # Prepare directories
    save_dir = f"./generated_samples/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    if args.compute_fid:
        # For FID computation, we need real images
        # the real images should be in 'real_img_dir'
        real_img_dir = args.real_img_dir
        if not os.path.exists(real_img_dir):
            raise FileNotFoundError(f"Real image directory {real_img_dir} not found.")
        # Generate samples
        total_samples = args.num_fid_samples
        iters_needed = (total_samples + args.batch_size - 1) // args.batch_size
        for i in range(iters_needed):
            current_batch_size = min(args.batch_size, total_samples - i * args.batch_size)
            x_t_1 = torch.randn(
                current_batch_size, args.num_channels, args.image_size, args.image_size
            ).to(device)
            fake_sample = sample_from_model(
                pos_coeff, netG, args.num_timesteps, x_t_1, T, args
            )
            fake_sample = to_range_0_1(fake_sample)
            for j, x in enumerate(fake_sample):
                index = i * args.batch_size + j
                torchvision.utils.save_image(
                    x, f'{save_dir}/{index}.png', normalize=False
                )
            print(f'Generated batch {i + 1}/{iters_needed}')
        # Compute FID
        paths = [save_dir, real_img_dir]
        fid = calculate_fid_given_paths(
            paths=paths, batch_size=50, device=device, dims=2048
        )
        print(f'FID = {fid}')
        
        if args.fid_output_path:
            output_dir = os.path.dirname(args.fid_output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(args.fid_output_path, 'w') as f:
                f.write(f'{fid}\n')
            #print(f'FID score saved to {args.fid_output_path}')
    
    else:
        # Generate and save sample images
        x_t_1 = torch.randn(
            args.batch_size, args.num_channels, args.image_size, args.image_size
        ).to(device)
        fake_sample = sample_from_model(
            pos_coeff, netG, args.num_timesteps, x_t_1, T, args
        )
        fake_sample = to_range_0_1(fake_sample)
        save_path = f'./samples_{args.dataset}.png'
        torchvision.utils.save_image(fake_sample, save_path, normalize=False)
        
        print(f'Sample images saved to {save_path}')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='DDGAN Testing Parameters')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed')
    
    parser.add_argument('--compute_fid', action='store_true', help='Compute FID score')
    parser.add_argument('--epoch_id', type=int, default=1000, help='Epoch ID to load checkpoint from')
    parser.add_argument('--real_img_dir', default='./real_images', help='Directory for real images (for FID computation)')
    parser.add_argument('--dataset', default='posluna', choices=['custom', 'posluna'], help='Dataset name')
    parser.add_argument('--exp', default='exp1', help='Experiment name')
    parser.add_argument('--num_fid_samples', type=int, default=5000, help='Number of samples to generate for FID computation')
    
    parser.add_argument('--fid_output_path', default='./fid_score.txt', help='Path to save the FID score')
    
    args = parser.parse_args()
    
    sample_and_test(args)

#cloner174
# 05 Oct 2024