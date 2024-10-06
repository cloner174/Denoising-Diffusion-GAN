import numpy as np
import random
import os
import shutil
import multiprocessing
import argparse
import json
import ast
import glob

from ddgan import main  # main function from training script

from additionals.utilities import load_json_to_dict, run_bash_command, find_python_command, \
    save_dict_to_json, modify_json_file

from additionals.images import simple_convert


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
    
    def update_velocity(self, global_best_position, c1, c2, w, max_velocity=None):
        
        for param in self.position:
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (self.best_position[param] - self.position[param])
            social_velocity = c2 * r2 * (global_best_position[param] - self.position[param])
            self.velocity[param] = w * self.velocity[param] + cognitive_velocity + social_velocity
            if max_velocity is not None:
                self.velocity[param] = max(-max_velocity, min(self.velocity[param], max_velocity))
    
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
    
    def __init__(self, search_space, num_particles=10, num_iterations=20, c1=1.5, c2=1.5, w=0.7, do_clamping=False):
        
        self.search_space = search_space
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        
        self.max_velocity = 1.0 if do_clamping else None
        
        self.particles = [Particle(search_space) for _ in range(num_particles)]
        self.global_best_position = {}
        self.global_best_score = float('inf')
        
        self.fid_min = float('inf')
        self.fid_max = float('-inf')
        self.loss_min = float('inf')
        self.loss_max = float('-inf')
    
    def optimize(self):
        
        prev_global_best_score = float('inf')
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}")
            scores = []
            positions = []
            
            if self.max_velocity is not None:
                self.w = 0.9 - iteration * (0.5 / self.num_iterations)  # Gradually decrease inertia weight
            
            args_list = [(particle.position, self.fid_min, self.fid_max, self.loss_min, self.loss_max) for particle in self.particles]
            # evaluate particles in parallel
            with multiprocessing.Pool(processes=min(self.num_particles, multiprocessing.cpu_count())) as pool:
                results = pool.starmap(evaluate, args_list)
            
            for i, (score, fid_score, loss_score) in enumerate(results):
                particle = self.particles[i]
                print(f"  Particle {i+1}/{self.num_particles}, Score: {score}")
                
                # Update min and max values
                self.fid_min = min(self.fid_min, fid_score)
                self.fid_max = max(self.fid_max, fid_score)
                self.loss_min = min(self.loss_min, loss_score)
                self.loss_max = max(self.loss_max, loss_score)
                
                # Update particle's best score and position
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
            
            # Update velocities and positions
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.c1, self.c2, self.w, self.max_velocity)
                particle.update_position(self.search_space)
            
            print(f"Global best score: {self.global_best_score}")
            print(f"Global best position: {self.global_best_position}")
            
            if iteration > 5 and abs(prev_global_best_score - self.global_best_score) < 1e-3:
                print("Stopping early due to minimal improvement in global best score.")
                break
            
            prev_global_best_score = self.global_best_score



def evaluate(hyperparams, fid_min, fid_max, loss_min, loss_max):
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    
    # hyperparameters from the particle's position
    config['lr_g'] = hyperparams['lr_g']
    config['lr_d'] = hyperparams['lr_d']
    config['batch_size'] = hyperparams['batch_size']
    config['nz'] = hyperparams['nz']
    config['ngf'] = hyperparams['ngf']
    config['t_emb_dim'] = hyperparams['t_emb_dim']
    config['beta1'] = hyperparams['beta1']
    config['beta2'] = hyperparams['beta2']
    
    # other training parameters
    config['num_epoch'] = 1  # Set epochs as needed
    config['exp'] = f"pso_eval_{random.randint(0, int(1e6))}"
    
    args = argparse.Namespace(**config)

    # Run the training
    main(args)
    
    exp_path = os.path.join("./saved_info/dd_gan", args.dataset, args.exp)
    
    # use FID score:
    
    fid_file = os.path.join(exp_path, 'fid_score.txt')
    
    temp_path = os.path.join(config['save_dir'], 'images_from_npy')
    limited_slices = config.get('limited_slices', False)
    npy_files = glob.glob(os.path.join(config['data_dir'], '*/*.npy'))
    os.makedirs(temp_path, exist_ok=True)
    for npy_file_path in npy_files:
        num_slices = 64
        num_skip = 7 if limited_slices else 1
        current_image = np.load(npy_file_path)
        for slice_index in range(1, min(num_slices, current_image.shape[0]), num_skip):
            try:
                simple_convert(npy_file_path, current_image[slice_index], temp_path, 'png', True)
            except:
                simple_convert(npy_file_path, current_image[slice_index], temp_path, 'png', False)
    
    # Run the test script to compute FID
    run_bash_command(f"{find_python_command()} test_ddgan.py --epoch_id {config['num_epoch']} --dataset {config['dataset']} --exp {config['exp']} --real_img_dir {temp_path} --compute_fid --fid_output_path {fid_file}")
    
    # Read FID score
    if os.path.exists(fid_file):
        with open(fid_file, 'r') as f:
            fid_str = f.readline().strip()
        try:
            fid_score = float(fid_str)
        except ValueError:
            fid_score = float('inf')
    else:
        fid_score = float('inf')
    
    # Read the final Generator loss
    loss_file = os.path.join(exp_path, 'final_loss.txt')
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            loss_str = f.readline().strip()
        try:
            loss_score = float(loss_str)
        except ValueError:
            loss_score = float('inf')
    else:
        loss_score = float('inf')
    
    fid_range = fid_max - fid_min if fid_max != fid_min else 1
    loss_range = loss_max - loss_min if loss_max != loss_min else 1
    
    normalized_fid = (fid_score - fid_min) / fid_range
    normalized_loss = (loss_score - loss_min) / loss_range
    
    score = normalized_fid + normalized_loss
    
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    
    return score, fid_score, loss_score


if __name__ == '__main__':
    
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser("PSO-GAN for LUNA16")
    
    parser.add_argument('--search_space', type=str, default='./configs/search_space_params.json', help='Path to JSON File for Search-Space Params')
    
    parser.add_argument('--config_file', type=str, default=None, help='Path to JSON File for configuration ddgan')
    
    parser.add_argument('--save_dir', type=str, default='./converted_images', help='Path to save images generated during FID score!')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_particles', type=int, default=10)
    parser.add_argument('--num_iterations', type=int, default=20)
    
    args = parser.parse_args()
    
    config = None
    if args.config_file is not None and os.path.isfile(args.config_file):
        try:
            config = load_json_to_dict(args.config_file, local=True)
            save_dict_to_json(config, filename='./configs/config.json', local=True)
            print(f"Config file is loaded from: {args.config_file}, and will be used!")
        except Exception as e:
            print(f"There was an error during loading your config file: {args.config_file}, Error: {e}")
            import warnings
            warnings.warn("This script will use the file './configs/config.json' for all configuration defaults!")
    
    if config is None:
        if not os.path.isfile('./configs/config.json'):
            run_bash_command(f"{find_python_command()} {os.curdir}/additionals/create_conf_default.py")
    
    config = load_json_to_dict('./configs/config.json', local=True)
    to_add = {'save_dir': args.save_dir}
    modify_json_file('./configs/config.json', to_add, local=True)
    
    config = load_json_to_dict('./configs/config.json', local=True)
    
    with open(args.search_space, 'r') as f:
        search_space = json.load(f)
    
    for key, val in search_space.items():
        if key == 'step':
            continue
        
        search_space[key] = ast.literal_eval(val)
    
    if 'step' not in search_space:
        search_space['step'] = {}
    
    search_space['step']['batch_size'] = args.batch_size
    
    pso = PSO(search_space, args.num_particles, args.num_iterations)
    
    pso.optimize()

#cloner174