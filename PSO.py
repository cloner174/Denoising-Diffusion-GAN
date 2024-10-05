import numpy as np
import random
import os
import shutil
import multiprocessing
from final import main  #main function from training script
import argparse
from conf_file import config


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
    
    def update_velocity(self, global_best_position, c1, c2, w, max_velocity = None):
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
    
    def __init__(self, search_space, num_particles=10, num_iterations=20, c1=1.5, c2=1.5, w=0.7, limited_iter = 'no', do_clamping = False):
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
        
        global limited_iter_
        limited_iter_ = limited_iter
    
    def optimize(self):
        if multiprocessing.get_start_method() != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}")
            scores = []
            positions = []
            
            if self.max_velocity is not None:
                self.w = 0.9 - iteration * (0.5 / self.num_iterations)  # Gradually decrease inertia weight
            
            # evaluate particles in parallel
            with multiprocessing.Pool(processes=min(self.num_particles, multiprocessing.cpu_count())) as pool:
                results = [pool.apply_async(evaluate, args=(particle.position,)) for particle in self.particles]
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
                particle.update_velocity(self.global_best_position, self.c1, self.c2, self.w, self.max_velocity)
                particle.update_position(self.search_space)
            print(f"Global best score: {self.global_best_score}")
            print(f"Global best position: {self.global_best_position}")
            
            if iteration > 5 and abs(self.global_best_score - scores[-1]) < 1e-3:
                print("Stopping early due to minimal improvement in global best score.")
                break

def evaluate(hyperparams):
    
    # multiple seeds for evaluation
    seeds = [42, 1337, 2023]
    
    scores = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        
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
        config['num_epoch'] = 1  # Increased epochs for better evaluation
        config['exp'] = f"pso_eval_{random.randint(0, 1e6)}"
        config['limited_iter'] = limited_iter_
    
        args = argparse.Namespace(**config)
        
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
        
        scores.append(score)
    
    return np.mean(scores)


#cloner174