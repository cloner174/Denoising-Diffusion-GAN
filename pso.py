# In the name of God
#
"""
PSO-GAN Hyperparameter Optimization Script

This script performs hyperparameter optimization for a GAN using Particle Swarm Optimization (PSO).
It evaluates different sets of hyperparameters by training the GAN and computing a combined score
based on the FID score and the generator loss.

- FID and loss scores
- multiprocessing
"""

import os
import sys
import random
import shutil
import argparse
import json
import ast
import multiprocessing
from typing import Dict, Tuple, List

import numpy as np
import torch
import logging
import subprocess

from ddgan import main, cleanup

from additionals.utilities import (
    load_json_to_dict,
    run_bash_command,
    find_python_command,
    save_dict_to_json,
    modify_json_file,
    install_package,
    load_slice_info
)

from additionals.images import nii_to_png


def setup_logger(log_file: str = 'pso_gan_optimization.log'):
    """
    Sets up the logger to output logs to both console and a file.
    
    Args:
        log_file (str): The filename for the log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Formatter to include time, level, and message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream handler (console)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


# Initialize logger
logger = setup_logger()


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures that CUDA selects the same algorithms each time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_bash_command(command: str) -> None:
    """
    Executes a bash command and logs its output.
    
    Args:
        command (str): The command to execute.
    
    Raises:
        RuntimeError: If the command execution fails.
    """
    logger.info(f"Executing command: {command}")
    try:
        # Execute the command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Ensures the output is string, not bytes
        )
        
        # Stream the output and error in real-time
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            if output:
                logger.info(output.strip())
            if error:
                logger.error(error.strip())
            if output == '' and error == '' and process.poll() is not None:
                break
        
        # Check for return code
        return_code = process.poll()
        if return_code != 0:
            raise RuntimeError(f"Command failed with return code {return_code}")
    
    except Exception as e:
        logger.error(f"An error occurred while executing the command: {e}")
        raise


class Particle:
    """
    Represents a particle in the PSO algorithm.
    """
    def __init__(self, search_space: Dict, seed: int = 42):
        """
        Initialize a Particle with random position and velocity within the search space.
        
        Args:
            search_space (dict): Dictionary defining the hyperparameter search space.
            seed (int): Seed for reproducibility.
        """
        self.seed = seed
        set_random_seeds(self.seed)
        
        self.position = {}
        self.velocity = {}
        self.best_position = {}
        self.best_score = float('inf')
        
        # Initialize position and velocity for each hyperparameter
        for param, bounds in search_space.items():
            if param == 'step':
                continue
            
            min_val, max_val = bounds
            if isinstance(min_val, int):
                step = search_space.get('step', {}).get(param, 1)
                possible_values = list(range(min_val, max_val + 1, step))
                self.position[param] = random.choice(possible_values)
                self.velocity[param] = random.uniform(-(max_val - min_val), max_val - min_val)
            else:
                self.position[param] = random.uniform(min_val, max_val)
                self.velocity[param] = random.uniform(-(max_val - min_val), max_val - min_val)
        
        self.best_position = self.position.copy()
    
    def update_velocity(self, global_best_position: Dict, c1: float, c2: float, w: float, max_velocity: float = None):
        """
        Update the particle's velocity based on its own experience and the swarm's best position.
        
        Args:
            global_best_position (dict): The best position found by the swarm.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
            w (float): Inertia weight.
            max_velocity (float, optional): Maximum allowed velocity for clamping.
        """
        for param in self.position:
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (self.best_position[param] - self.position[param])
            social_velocity = c2 * r2 * (global_best_position[param] - self.position[param])
            self.velocity[param] = w * self.velocity[param] + cognitive_velocity + social_velocity
            
            # Apply velocity clamping if specified
            if max_velocity is not None:
                self.velocity[param] = max(-max_velocity, min(self.velocity[param], max_velocity))
    
    def update_position(self, search_space: Dict):
        """
        Update the particle's position based on its velocity and enforce search space bounds.

        Args:
            search_space (dict): Dictionary defining the hyperparameter search space.
        """
        for param in self.position:
            self.position[param] += self.velocity[param]
            # Enforce bounds and discretize if necessary
            min_val, max_val = search_space[param]
            if isinstance(min_val, int):
                step = search_space.get('step', {}).get(param, 1)
                self.position[param] = int(round(self.position[param] / step) * step)
                self.position[param] = max(min_val, min(self.position[param], max_val))
            else:
                self.position[param] = max(min_val, min(self.position[param], max_val))


class PSO:
    """
    Particle Swarm Optimization class for optimizing hyperparameters.
    """
    
    def __init__(
        self,
        search_space: Dict,
        num_particles: int = 10,
        num_iterations: int = 20,
        c1: float = 1.5,
        c2: float = 1.5,
        w: float = 0.7,
        do_clamping: bool = False,
        use_multiprocessing: bool = False,
        seed: int = 42
    ):
        """
        Initialize the PSO optimizer.
        Args:
            search_space (dict): Dictionary defining the hyperparameter search space.
            num_particles (int): Number of particles in the swarm.
            num_iterations (int): Number of iterations to perform.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
            w (float): Inertia weight.
            do_clamping (bool): Whether to clamp velocities.
            use_multiprocessing (bool): Whether to use multiprocessing for evaluation.
            seed (int): Seed for reproducibility.
        """
        
        self.search_space = search_space
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        
        self.use_multiprocessing = use_multiprocessing
        self.max_velocity = 1.0 if do_clamping else None
        
        self.seed = seed
        set_random_seeds(self.seed)
        
        # Initialize particles with unique seeds
        self.particles = [Particle(search_space, seed=self.seed + i) for i in range(num_particles)]
        self.global_best_position = self.particles[0].position.copy()
        self.global_best_score = float('inf')
    
    def optimize(self):
        """
        Run the PSO optimization process.
        """
        prev_global_best_score = float('inf')
        for iteration in range(self.num_iterations):
            
            logger.info(f"Iteration {iteration + 1}/{self.num_iterations}")
            if self.max_velocity is not None:
                # Gradually decrease inertia weight (Linear Decreasing Inertia Weight)
                self.w = 0.9 - iteration * (0.5 / self.num_iterations)
                self.w = max(self.w, 0.4)  # Ensure w does not go below 0.4
                logger.info(f"Updated inertia weight: {self.w:.4f}")
            
            # Arguments for evaluation
            positions = [particle.position for particle in self.particles]
            
            # Assign unique seeds for each evaluation
            seeds = [self.seed + i + iteration * self.num_particles for i in range(self.num_particles)]
            
            # Evaluate particles
            if self.use_multiprocessing:
                with multiprocessing.Pool(processes=min(self.num_particles, multiprocessing.cpu_count())) as pool:
                    results = pool.starmap(evaluate_wrapper, zip(positions, seeds))
            else:
                results = []
                for i, (position, seed) in enumerate(zip(positions, seeds)):
                    logger.info(f"Evaluating particle {i + 1}/{self.num_particles}")
                    result = evaluate(position, seed=seed)
                    results.append(result)
            
            # Update particles
            for i, particle in enumerate(self.particles):
                
                score = results[i]
                logger.info(f"Particle {i + 1}/{self.num_particles}, Score: {score}")
                
                # Update particle's best score and position
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                    logger.info(f"Particle {i + 1} found a new best position.")
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
                    logger.info(f"Global best updated by particle {i + 1}.")
            
            # Update velocities and positions
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.c1, self.c2, self.w, self.max_velocity)
                particle.update_position(self.search_space)
            
            logger.info(f"Global best score: {self.global_best_score}")
            logger.info(f"Global best position: {self.global_best_position}")
            
            # Early stopping if improvement is minimal
            if iteration > 5 and abs(prev_global_best_score - self.global_best_score) < 1e-3:
                logger.info("Stopping early due to minimal improvement in global best score.")
                break
            
            prev_global_best_score = self.global_best_score


def evaluate_wrapper(hyperparams: Dict, seed: int) -> float:
    """
    Wrapper function for multiprocessing to evaluate hyperparameters.
    
    Args:
        hyperparams (dict): Hyperparameters to evaluate.
        seed (int): Seed for reproducibility.
    
    Returns:
        float: Combined normalized score.
    """
    return evaluate(hyperparams, seed=seed)


def evaluate(hyperparams: Dict, seed: int) -> float:
    """
    Evaluate the given hyperparameters by training the GAN and computing the score.
    
    Args:
        hyperparams (dict): Hyperparameters to evaluate.
        seed (int): Seed for reproducibility.
    
    Returns:
        float: Combined normalized score (lower is better).
    """
    unique_id = random.randint(0, int(1e6))
    base_config_path = './configs/config.json'
    
    # configuration
    config_path, config = prepare_config(base_config_path, hyperparams, unique_id)
    exp_path = os.path.join("./saved_info/dd_gan", config['dataset'], config['exp'])
    
    try:
        # Set seeds for reproducibility
        set_random_seeds(seed)
        
        # training
        run_training(config_path)
        
        # loss
        loss_score = compute_loss(exp_path)
        
        # FID score
        if config.get('with_FID', False):
            fid_score = compute_fid_score(config, unique_id)
        else:
            fid_score = 0.0
        
        # normalize scores
        normalized_loss = normalize_score(loss_score, config.get('loss_min', 0), config.get('loss_max', 1))
        normalized_fid = normalize_score(fid_score, config.get('fid_min', 0), config.get('fid_max', 300))
        
        # weight the scores
        loss_weight = 0.5
        fid_weight = 0.5
        score = (loss_weight * normalized_loss) + (fid_weight * normalized_fid)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        score = float('inf')
    
    finally:
        # clean up temporary files
        cleanup_experiment(config, unique_id)
    
    return score


def prepare_config(base_config_path: str, hyperparams: Dict, unique_id: int) -> Tuple[str, Dict]:
    """
    Prepare the configuration by updating it with the given hyperparameters.
    
    Args:
        base_config_path (str): Path to the base configuration file.
        hyperparams (dict): Hyperparameters to update.
        unique_id (int): Unique identifier for this run.
    
    Returns:
        tuple: (new_config_path, updated_config)
    """
    config = load_json_to_dict(base_config_path, local=True)
    config.update(hyperparams)
    config['exp'] = f"pso_eval_{unique_id}"
    config['num_epoch'] = 1  # Set epochs to 1 for quick evaluation
    config['seed'] = config.get('seed', 42)  # Ensure seed is present
    
    new_config_path = f'./configs/config_{unique_id}.json'
    save_dict_to_json(config, new_config_path, local=True)
    
    return new_config_path, config


def run_training(config_path: str):
    """
    Run the training script with the provided configuration.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Raises:
        RuntimeError: If the training script fails.
    """
    
    command = f"{sys.executable} train_ddgan.py --use_config_file True --config_file {config_path}"
    try:
        run_bash_command(command)
    
    except Exception as e:
        raise RuntimeError(f"Training failed with error: {e}")


def compute_loss(exp_path: str) -> float:
    """
    Compute the generator loss from the training output.
    
    Args:
        exp_path (str): Path to the experiment directory.
    
    Returns:
        float: Generator loss.
    """
    loss_file = os.path.join(exp_path, 'final_loss.txt')
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            loss_score = float(f.readline().strip())
    else:
        loss_score = float('inf')
    
    return loss_score


def compute_fid_score(config: Dict, unique_id: int) -> float:
    """
    Compute the FID score using the test script.
    
    Args:
        config (dict): Configuration dictionary.
        unique_id (int): Unique identifier for this run.
    
    Returns:
        float: FID score.
    """
    # real images directory has enough images
    real_img_dir = os.path.join(config['save_dir'], 'real_images')
    if not os.path.isdir(real_img_dir) or len(os.listdir(real_img_dir)) < 100:
        if config.get('path_to_slices_info'):
            slices_info = load_slice_info(config['path_to_slices_info'])
            nii_to_png(slices_info, save_dir=real_img_dir, lim=1000)
        else:
            raise FileNotFoundError("Path to slices info is not specified in the config.")
    
    fid_file = os.path.join('./saved_info/', f'fid_score_{unique_id}.txt')
    
    command = (
        f"{sys.executable} test_ddgan.py --epoch_id {config['num_epoch']} "
        f"--dataset {config['dataset']} --exp {config['exp']} "
        f"--real_img_dir {real_img_dir} --compute_fid --fid_output_path {fid_file}"
    )
    
    try:
        run_bash_command(command)
    
    except Exception as e:
        raise RuntimeError(f"FID computation failed with error: {e}")
    
    if os.path.exists(fid_file):
        with open(fid_file, 'r') as f:
            fid_score = float(f.readline().strip())
    else:
        fid_score = float('inf')  # Assign a high value if FID score is not found
    
    return fid_score


def normalize_score(score: float, score_min: float, score_max: float) -> float:
    """
    Normalize a score to the range [0, 1].
    
    Args:
        score (float): The score to normalize.
        score_min (float): Minimum possible score.
        score_max (float): Maximum possible score.
    
    Returns:
        float: Normalized score.
    """
    if score_max == score_min:
        return 0.0
    
    normalized = (score - score_min) / (score_max - score_min)
    
    # normalized score within [0, 1]
    return max(0.0, min(1.0, normalized))


def cleanup_experiment(config: Dict, unique_id: int):
    """
    Clean up temporary files and directories created during the evaluation.
    
    Args:
        config (dict): Configuration dictionary.
        unique_id (int): Unique identifier for this run.
    """
    
    exp_path = os.path.join("./saved_info/dd_gan", config['dataset'], config['exp'])
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)
    
    generated_samples_dir = os.path.join(config['save_dir'], 'generated_samples')
    if os.path.exists(generated_samples_dir):
        shutil.rmtree(generated_samples_dir)
    
    #real_img_dir = os.path.join(config['save_dir'], 'real_images')
    #if os.path.exists(real_img_dir):
    #    shutil.rmtree(real_img_dir)
    
    temp_config_path = f'./configs/config_{unique_id}.json'
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    
    fid_file = os.path.join('./saved_info/', f'fid_score_{unique_id}.txt')
    if os.path.exists(fid_file):
        os.remove(fid_file)


def install_ninja():
    """
    Ensure that the 'ninja' package is installed.
    """
    try:
        import ninja
    except ModuleNotFoundError:
        install_package('ninja')


def main():
    """
    Main function to parse arguments and run the PSO optimizer.
    """
    install_ninja()
    
    parser = argparse.ArgumentParser("PSO-GAN for LUNA16")
    
    parser.add_argument('--search_space', type=str, default='./configs/search_space_params.json',
                        help='Path to JSON file for search-space parameters')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to JSON file for DDGAN configuration')
    parser.add_argument('--save_dir', type=str, default='./converted_images',
                        help='Path to save images generated during FID score computation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Initial batch size to use (step size is defined in search_space_params.json)')
    parser.add_argument('--num_particles', type=int, default=10,
                        help='Number of particles in the swarm')
    parser.add_argument('--num_iterations', type=int, default=20,
                        help='Number of iterations to perform')
    parser.add_argument('--limited_iteration_mode', type=int, default=202,
                        help='Limited iteration mode parameter')
    parser.add_argument('--with_FID', action='store_true', help='Compute FID score during evaluation')
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
    parser.add_argument('--use_multiprocessing', action='store_true', help='Use multiprocessing during evaluation')
    parser.add_argument('--log_file', type=str, default='pso_gan_optimization.log',
                        help='Path to the log file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Reconfigure logger if a different log file is specified
    if args.log_file != 'pso_gan_optimization.log':
        logger.handlers.clear()
        logger = setup_logger(log_file=args.log_file)
    
    if args.use_multiprocessing:
        logger.info("Starting with multiprocessing")
        multiprocessing.set_start_method('spawn', force=True)
    
    # Set the global seed
    set_random_seeds(args.seed)
    
    # load or create configuration
    if args.config_file and os.path.isfile(args.config_file):
        
        config = load_json_to_dict(args.config_file, local=True)
        save_dict_to_json(config, filename='./configs/config.json', local=True)
        logger.info(f"Config file loaded from: {args.config_file}")
    else:
        if not os.path.isfile('./configs/config.json'):
            logger.info("No config file provided. Using default configuration.")
            run_bash_command(f"{find_python_command()} {os.curdir}/additionals/create_conf_default.py")
        
        config = load_json_to_dict('./configs/config.json', local=True)
    
    # update base configuration with additionals
    to_add = {
        'save_dir': args.save_dir,
        'limited_iter': args.limited_iteration_mode,
        'resume': args.resume,
        'num_workers': 0,
        'with_FID': args.with_FID,
        'seed': args.seed  # Ensure seed is included
    }
    modify_json_file('./configs/config.json', to_add, local=True)
    
    # load the search space
    with open(args.search_space, 'r') as f:
        search_space = json.load(f)
    
    # Adjust search space parsing
    # Remove ast.literal_eval since search_space now contains lists
    if 'step' not in search_space:
        search_space['step'] = {}
    
    search_space['step']['batch_size'] = args.batch_size
    
    # initialize PSO
    pso = PSO(
        search_space=search_space,
        num_particles=args.num_particles,
        num_iterations=args.num_iterations,
        c1=1.5,
        c2=1.5,
        w=0.7,
        do_clamping=True,  # Enable clamping if desired
        use_multiprocessing=args.use_multiprocessing,
        seed=args.seed
    )
    
    # run optimization
    pso.optimize()
    
    # save best hyperparameters found
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(pso.global_best_position, f, indent=4)
    
    logger.info("Optimization completed.")
    logger.info("Best hyperparameters found:")
    logger.info(json.dumps(pso.global_best_position, indent=4))


if __name__ == '__main__':
    main()
