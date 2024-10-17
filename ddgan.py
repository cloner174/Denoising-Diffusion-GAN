import argparse
import os
import sys
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.nn.functional as F

from datasets_prep.custom import DatasetCustom, PositivePatchDataset, Luna16Dataset
from score_sde.models.discriminator import Discriminator_small, Discriminator_large
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from ema import EMA


def copy_source(file, output_dir):
    """Copy the source file to the output directory."""
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    """Broadcast parameters from the root process to all other processes."""
    for param in params:
        dist.broadcast(param.data, src=0)


# Diffusion coefficients functions

def var_func_vp(t, beta_min, beta_max):
    """Variance function for VP SDE."""
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    """Variance function for geometric progression."""
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input_tensor, t, shape):
    """Extract values from a tensor based on time indices."""
    out = torch.gather(input_tensor, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def get_time_schedule(args, device):
    """Get the time schedule tensor."""
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    """Get the sigma schedule for the diffusion process."""
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class DiffusionCoefficients:
    """Class to compute and store diffusion coefficients."""

    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, noise=None):
    """Diffuse the data for time step t."""
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """Generate a pair of disturbed images for training."""
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise
    return x_t, x_t_plus_one


# Posterior sampling functions

class PosteriorCoefficients:
    """Class to compute and store posterior coefficients."""

    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    """Sample from the posterior distribution."""
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
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)
    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    """Generate samples from the model by reverse diffusion."""
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
    return x


def train(rank, gpu, args):
    """Training loop for the DDGAN model."""
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    nz = args.nz  # Latent dimension
    
    exp = args.exp
    
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)
    
    exp_path = os.path.join(parent_dir,exp)
    
    # Define transformations
    transform_list = []
    if hasattr(args, 'do_resize') and args.do_resize.lower() == 'yes':
        transform_list.append(transforms.Resize(args.image_size))
    if hasattr(args, 'to_tensor_transform') and args.to_tensor_transform.lower() == 'yes':
        transform_list.append(transforms.ToTensor())
    if hasattr(args, 'use_normalize') and args.use_normalize.lower() == 'yes':
        normalize_stat = transforms.Normalize(
            (0.5,) * args.num_channels, (0.5,) * args.num_channels)
        transform_list.append(normalize_stat)
    if hasattr(args, 'CenterCrop') and args.CenterCrop.lower() == 'yes':
        transform_list.append(transforms.CenterCrop(args.image_size))

    if transform_list:
        transform = transforms.Compose(transform_list)
    else:
        transform = None

    # Load dataset
    if args.dataset == 'custom':
        dataset = DatasetCustom(
            data_dir=args.data_dir, class_=args.mode, transform=transform)
    elif args.dataset == 'posluna':
        dataset = PositivePatchDataset(
            data_dir=args.data_dir, transform=transform, limited_slices=args.limited_slices)
    elif args.dataset == 'luna16':
        bound_exp_lim = getattr(args, 'bound_expand_limit', 1 if args.limited_slices else 5)
        path_to_slices_info = getattr(args, 'path_to_slices_info', None)
        dataset = Luna16Dataset(
            data_dir=args.data_dir,
            mask_dir=args.mask_dir,
            transform=transform,
            bound_exp_lim=bound_exp_lim,
            path_to_slices_info=path_to_slices_info,
            _3d=getattr(args, 'use_3d_mode', False),
            bounders=args.num_channels,
            single_axis=args.limited_slices,
            _where=args.axis_for_limit)

    # Data loader and sampler
    try:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True)
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True)
    except Exception as e:
        print('An error occurred when attempting to use the data loader:', e)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True)

    if hasattr(dataset, 'limited_slices') and dataset.limited_slices:
        warnings.warn(f"The Limited-Slices mode is On! Dataset size: {len(dataset)}")

    netG = NCSNpp(args).to(device)
    
    if args.disc_small.lower() == 'yes':
        netD = Discriminator_small(
            nc=2 * args.num_channels,
            ngf=args.ngf,
            t_emb_dim=args.t_emb_dim,
            act=nn.LeakyReLU(0.2)
        ).to(device)
    
    else:
        netD = Discriminator_large(
            nc=2 * args.num_channels,
            ngf=args.ngf,
            t_emb_dim=args.t_emb_dim,
            act=nn.LeakyReLU(0.2)
        ).to(device)
    
    if args.distributed:
        broadcast_params(netG.parameters())
        broadcast_params(netD.parameters())
    
    optimizerD = optim.Adam(
        netD.parameters(), 
        lr=args.lr_d, 
        betas=(args.beta1_d, args.beta2_d),
        weight_decay=args.weight_decay_D
        )
    
    optimizerG = optim.Adam(
        netG.parameters(), 
        lr=args.lr_g, 
        betas=(args.beta1_g, args.beta2_g),
        weight_decay=args.weight_decay_G
        )
    
    if args.use_ema:
        emaG = EMA(netG, optimizerG, ema_decay=args.ema_decay, device=device)
    
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)
    # Distributed Data Parallel
    
    if args.distributed:
        netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
        netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])
    
    coeff = DiffusionCoefficients(args, device)
    pos_coeff = PosteriorCoefficients(args, device)
    T = get_time_schedule(args, device)
    # Load checkpoint
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

        def adjust_state_dict(state_dict):
            """Remove 'module.' prefix from state dict keys if present."""
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_k = k[len('module.'):]
                else:
                    new_k = k
                new_state_dict[new_k] = v
            return new_state_dict

        # Adjust state_dict keys if necessary
        if args.distributed:
            netG_state_dict = checkpoint['netG_dict']
            netD_state_dict = checkpoint['netD_dict']
        else:
            netG_state_dict = adjust_state_dict(checkpoint['netG_dict'])
            netD_state_dict = adjust_state_dict(checkpoint['netD_dict'])

        netG.load_state_dict(netG_state_dict)
        netD.load_state_dict(netD_state_dict)

        # Load optimizers and schedulers
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        init_epoch = checkpoint['epoch']
        if args.use_ema and 'emaG' in checkpoint:
            emaG.load_state_dict(checkpoint['emaG'])
        print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    # Limited iteration mode
    limited_iter = None
    if hasattr(args, 'limited_iter'):
        print(f"Limited Iteration Mode is On! -> {args.limited_iter}")
        if isinstance(args.limited_iter, int):
            limited_iter = [i for i in range(args.limited_iter)]
            print(f"Limiting to {args.limited_iter} iterations per epoch.")
        elif isinstance(args.limited_iter, list):
            limited_iter = [i for i in range(int(np.mean(args.limited_iter)))]
            print(f"Limiting to {int(np.mean(args.limited_iter))} iterations per epoch.")
        print('-----------------')

    # Training loop
    for epoch in range(init_epoch, args.num_epoch + 1):
        if args.distributed and 'train_sampler' in locals():
            train_sampler.set_epoch(epoch)

        for iteration, (x, _) in enumerate(data_loader):
            if limited_iter is not None and iteration not in limited_iter:
                break

            # Train Discriminator
            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()

            real_data = x.to(device, non_blocking=True)
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # Real samples
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            errD_real = F.softplus(-D_real).mean()
            errD_real.backward(retain_graph=True)

            # Gradient penalty
            if args.lazy_reg is None or global_step % args.lazy_reg == 0:
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()

            # Fake samples
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errD_fake = F.softplus(output).mean()
            errD_fake.backward()
            errD = errD_real + errD_fake
            
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=args.grad_clip_norm)
            
            optimizerD.step()

            # Train Generator
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errG = F.softplus(-output).mean()
            
            errG.backward()
            
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=args.grad_clip_norm)
            
            optimizerG.step()
            
            if args.use_ema:
                emaG.step()
            
            global_step += 1
            if iteration % 100 == 0 and rank == 0:
                print(f'Epoch {epoch + 1}, Iteration {iteration}, G Loss: {errG.item():.8f}, D Loss: {errD.item():.8f}')
        
        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()
        
        # Save outputs and checkpoints
        if rank == 0:
            # Save model state dictionaries appropriately
            if args.distributed:
                netG_state_dict = netG.module.state_dict()
                netD_state_dict = netD.module.state_dict()
            else:
                netG_state_dict = netG.state_dict()
                netD_state_dict = netD.state_dict()

            content = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'args': vars(args),
                'netG_dict': netG_state_dict,
                'optimizerG': optimizerG.state_dict(),
                'schedulerG': schedulerG.state_dict(),
                'netD_dict': netD_state_dict,
                'optimizerD': optimizerD.state_dict(),
                'schedulerD': schedulerD.state_dict(),
            }
            if args.use_ema:
                content['emaG'] = emaG.state_dict()
            torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    emaG.swap_parameters_with_ema(store_params_in_ema=True)
                torch.save(netG.state_dict(), os.path.join(exp_path, f'netG_{epoch}.pth'))
                if args.use_ema:
                    emaG.swap_parameters_with_ema(store_params_in_ema=True)

            # Save final generator loss
            with open(os.path.join(exp_path, 'final_loss.txt'), 'w') as f:
                f.write(f"{errG.item()}\n")


def init_processes(rank, size, fn, args):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'

    if args.distributed:
        dist.init_process_group(
            backend=getattr(args, 'what_backend', 'nccl'),
            init_method='env://',
            rank=rank,
            world_size=size
        )
        torch.cuda.set_device(args.local_rank)
        gpu = args.local_rank
        fn(rank, gpu, args)
        dist.barrier()
        cleanup()
    else:
        gpu = args.local_rank
        fn(rank, gpu, args)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def main(args):
    
    # Set default values or load from config
    args.num_proc_node = 1
    args.num_process_per_node = 1
    args.node_rank = 0
    args.master_address = '127.0.0.1'
    
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    
    if size > 1 and args.distributed:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print(f'Node rank {args.node_rank}, local proc {rank}, global proc {global_rank}')
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        print('Starting in non-distributed mode')
        args.local_rank = 0
        init_processes(0, size, train, args)

