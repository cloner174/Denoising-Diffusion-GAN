import numpy as np
import random
import os
import shutil
import multiprocessing
from final_script import main  #main function from training script

class Args:
    pass

class Particle:
    def __init__(self, search_space):
        self.position = {}
        self.velocity = {}
        self.best_position = {}
        self.best_score = float('inf')
        
        # position and velocity
        for param in search_space:
            if param == 'step':
                continue
            min_val, max_val = search_space[param]
            if isinstance(min_val, int):
                step = search_space.get('step', {}).get(param, 1)
                possible_values = list(range(min_val, max_val + 1, step))
                self.position[param] = random.choice(possible_values)
                self.velocity[param] = random.uniform(-(max_val - min_val), max_val - min_val)
            else:
                self.position[param] = random.uniform(min_val, max_val)
                self.velocity[param] = random.uniform(-(max_val - min_val), max_val - min_val)
        
        self.best_position = self.position.copy()
    
    def update_velocity(self, global_best_position, c1, c2, w):
        for param in self.position:
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (self.best_position[param] - self.position[param])
            social_velocity = c2 * r2 * (global_best_position[param] - self.position[param])
            self.velocity[param] = w * self.velocity[param] + cognitive_velocity + social_velocity
    
    def update_position(self, search_space):
        for param in self.position:
            self.position[param] += self.velocity[param]
            # bounds
            min_val, max_val = search_space[param]
            if isinstance(min_val, int):
                step = search_space.get('step', {}).get(param, 1)
                self.position[param] = int(round(self.position[param] / step) * step)
                self.position[param] = max(min_val, min(self.position[param], max_val))
            else:
                self.position[param] = max(min_val, min(self.position[param], max_val))
    
    

class PSO:
    
    def __init__(self, search_space, num_particles=10, num_iterations=20, c1=1.5, c2=1.5, w=0.7):
        self.search_space = search_space
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.particles = [Particle(search_space) for _ in range(num_particles)]
        self.global_best_position = {}
        self.global_best_score = float('inf')
    
    def optimize(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}")
            scores = []
            positions = []
            # evaluate particles in parallel
            pool = multiprocessing.Pool(processes=min(self.num_particles, multiprocessing.cpu_count()))
            results = []
            for particle in self.particles:
                positions.append(particle.position.copy())
                result = pool.apply_async(evaluate, args=(particle.position,))
                results.append(result)
            pool.close()
            pool.join()
            for i, result in enumerate(results):
                score = result.get()
                particle = self.particles[i]
                print(f"  Particle {i+1}/{self.num_particles}, Score: {score}")
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
            # Update velocities and positions
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.c1, self.c2, self.w)
                particle.update_position(self.search_space)
            print(f"Global best score: {self.global_best_score}")
            print(f"Global best position: {self.global_best_position}")
        
    
def evaluate(hyperparams):
    args = Args()
    # default arguments
    args.seed = 1024
    args.resume = False
    args.num_workers = 2
    args.mode = 'train'
    args.disc_small = 'yes'
    args.data_dir = '/content/Drive/MyDrive/cloner174/Luna16/data/Slices'
    args.dataset = 'custom'
    args.image_size = 64
    args.num_channels = 3
    args.centered = True
    args.use_geometric = False
    args.beta_min = 0.1
    args.beta_max = 20.0
    args.num_channels_dae = 128
    args.n_mlp = 4
    args.ch_mult = [1, 2, 2, 2]
    args.num_res_blocks = 2
    args.attn_resolutions = (16,)
    args.dropout = 0.0
    args.resamp_with_conv = True
    args.conditional = True
    args.fir = True
    args.fir_kernel = [1, 3, 3, 1]
    args.skip_rescale = True
    args.resblock_type = 'biggan'
    args.progressive = 'none'
    args.progressive_input = 'residual'
    args.progressive_combine = 'sum'
    args.embedding_type = 'positional'
    args.fourier_scale = 16.0
    args.not_use_tanh = False
    args.z_emb_dim = 256
    args.num_timesteps = 1
    args.no_lr_decay = False
    args.use_ema = True
    args.ema_decay = 0.9999
    args.r1_gamma = 0.02
    args.lazy_reg = 15
    args.num_proc_node = 1
    args.num_process_per_node = 1
    args.node_rank = 0
    args.local_rank = 0
    args.master_address = '127.0.0.1'
    args.save_content = False
    # hyperparameters from the particle's position
    args.lr_g = hyperparams['lr_g']
    args.lr_d = hyperparams['lr_d']
    args.batch_size = hyperparams['batch_size']
    args.nz = hyperparams['nz']
    args.ngf = hyperparams['ngf']
    args.t_emb_dim = hyperparams['t_emb_dim']
    args.beta1 = hyperparams['beta1']
    args.beta2 = hyperparams['beta2']
    # other training parameters
    args.num_epoch = 5  # Increased epochs for better evaluation
    args.exp = f"pso_eval_{random.randint(0, 1e6)}"
    # Run the training
    main(args)
    # the final Generator loss
    exp_path = os.path.join("./saved_info/dd_gan", args.dataset, args.exp)
    loss_file = os.path.join(exp_path, 'final_loss.txt')
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            loss_str = f.readline().strip()
        try:
            score = float(loss_str)
        except ValueError:
            score = float('inf')
    else:
        score = float('inf')
    # clean up experiment directory to save space
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)

    return score

if __name__ == '__main__':
    search_space = {
        'lr_g': (1e-6, 1e-3),
        'lr_d': (1e-6, 1e-3),
        'batch_size': (32, 256),
        'nz': (50, 200),
        'ngf': (32, 128),
        't_emb_dim': (128, 512),
        'beta1': (0.3, 0.9),
        'beta2': (0.7, 0.999),
        # step sizes for integer parameters
        'step': {
            'batch_size': 32,
            'nz': 10,
            'ngf': 16,
            't_emb_dim': 32,
        }
    }
    
    
    pso = PSO(search_space, num_particles=10, num_iterations=20)
    
    pso.optimize()
