# In the name of God

"""
GA-GAN Hyperparameter Optimization Script

- FID and/or Loss Scores
- Multiprocessing optional
"""

import os
import sys
import random
import shutil
import argparse
import json
import multiprocessing
from typing import Dict, Tuple, List

import numpy as np
import torch
import logging
import subprocess

#from ddgan import main, cleanup

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


def setup_logger(log_file: str = 'ga_gan_optimization.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = setup_logger()


def set_random_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    
    config_path, config = prepare_config(base_config_path, hyperparams, unique_id)
    exp_path = os.path.join("./saved_info/dd_gan", config['dataset'], config['exp'])
    os.makedirs(exp_path, exist_ok=True)
    
    try:
        set_random_seeds(seed)
        
        run_training(config_path)
        loss_score = compute_loss(exp_path)
        
        if config.get('with_FID', False):
            fid_score = compute_fid_score(config, unique_id)
        else:
            fid_score = 0.0
        
        normalized_loss = normalize_score(loss_score, config.get('loss_min', 0), config.get('loss_max', 1))
        normalized_fid  = normalize_score(fid_score, config.get('fid_min', 0), config.get('fid_max', 300))
        
        loss_weight = 0  # 0.5 for half weighting
        fid_weight  = 1  # 0.5 for half weighting
        score = (loss_weight * normalized_loss) + (fid_weight * normalized_fid)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        score = float('inf')
    
    finally:
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
    config['exp'] = f"ga_eval_{unique_id}"
    config['num_epoch'] = 4
    # config['limited_iter'] = 202
    config['seed'] = config.get('seed', 42) 
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
    real_img_dir = os.path.join(config['save_dir'], 'CancerImagesTest')
    generated_samples_dir = os.path.join(config['save_dir'], f"generated_samples_{config['exp']}")
    os.makedirs(generated_samples_dir, exist_ok=True)
    
    if not os.path.isdir(real_img_dir) or len(os.listdir(real_img_dir)) < 100:
        if config.get('path_to_slices_info'):
            slices_info = load_slice_info(config['path_to_slices_info'])
            nii_to_png(
                slices_info, 
                save_dir=real_img_dir,
                lim=1000,
                do_resize_to=(int(config['image_size']), int(config['image_size']))
            )
        else:
            raise FileNotFoundError("Path to slices info is not specified in the config.")
    
    fid_file = os.path.join('./saved_info/', f'fid_score_{unique_id}.txt')
    
    command = (
        f"{sys.executable} test_ddgan.py --epoch_id {config['num_epoch']} "
        f"--generated_samples_dir {generated_samples_dir} "
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
        fid_score = float('inf')
    
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
    
    generated_samples_dir = os.path.join(config['save_dir'], f"generated_samples_{config['exp']}")
    if os.path.exists(generated_samples_dir):
        shutil.rmtree(generated_samples_dir)
    
    temp_config_path = f'./configs/config_{unique_id}.json'
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    
    fid_file = os.path.join('./saved_info/', f'fid_score_{unique_id}.txt')
    if os.path.exists(fid_file):
        os.remove(fid_file)

#####################
# GA Implementation #
#####################
class Chromosome:
    """
    Represents a set of hyperparameters for the GA.
    """
    def __init__(self, search_space: Dict, seed: int = 42):
        set_random_seeds(seed)
        self.hyperparams = {}
        self.fitness = float('inf')  # Lower is better here
        
        # Randomly initialize hyperparams based on search_space
        for param, bounds in search_space.items():
            if param == 'step':
                continue
            
            min_val, max_val = bounds
            if isinstance(min_val, int):
                step = search_space.get('step', {}).get(param, 1)
                possible_values = list(range(min_val, max_val + 1, step))
                self.hyperparams[param] = random.choice(possible_values)
            else:
                self.hyperparams[param] = random.uniform(min_val, max_val)
    
    def copy(self):
        """
        Returns a deep copy of this chromosome.
        """
        new_chrom = Chromosome({}, seed=0)
        new_chrom.hyperparams = self.hyperparams.copy()
        new_chrom.fitness = self.fitness
        return new_chrom


class GeneticAlgorithm:
    """
    Genetic Algorithm for hyperparameter optimization.
    """
    def __init__(
        self,
        search_space: Dict,
        population_size: int = 10,
        num_generations: int = 20,
        p_crossover: float = 0.8,
        p_mutation: float = 0.1,
        tournament_size: int = 2,
        use_multiprocessing: bool = False,
        seed: int = 42
    ):
        """
        Initialize GA hyperparameters.
        
        Args:
            search_space (dict): Hyperparameter search space.
            population_size (int): Number of chromosomes in the population.
            num_generations (int): How many generations to run.
            p_crossover (float): Probability of crossover between two parents.
            p_mutation (float): Probability of mutation for any gene.
            tournament_size (int): Tournament size for selection.
            use_multiprocessing (bool): Use multiprocessing for evaluation.
            seed (int): Random seed for reproducibility.
        """
        self.search_space = search_space
        self.population_size = population_size
        self.num_generations = num_generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.tournament_size = tournament_size
        self.use_multiprocessing = use_multiprocessing
        self.seed = seed
        
        set_random_seeds(self.seed)
        
        self.population: List[Chromosome] = [
            Chromosome(search_space, seed=self.seed + i) for i in range(self.population_size)
        ]
        
        self.best_chromosome = None
        self.best_fitness = float('inf')
    
    
    def optimize(self):
        """
        Main GA loop to optimize the hyperparameters.
        """
        logger.info("Starting Genetic Algorithm Optimization...")
        for gen in range(self.num_generations):
            logger.info(f"\n=== Generation {gen+1}/{self.num_generations} ===")
            
            self.evaluate_population()
            
            logger.info(f"Best fitness so far: {self.best_fitness}")
            logger.info(f"Best hyperparams so far: {self.best_chromosome.hyperparams if self.best_chromosome else {}}")
            
            # create next generation
            next_population: List[Chromosome] = []
            
            # If you want elitism, you can keep the best individual
            # For now, let's skip explicit elitism. If you want, you can do:
            # next_population.append(self.best_chromosome.copy())
            
            # Fill the rest of the new population
            while len(next_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                self.mutate(child1)
                self.mutate(child2)
                
                # Add children to next population
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
            
            # Replace old population with new one
            self.population = next_population
        
        # Final evaluation (to ensure the final population is also assessed)
        self.evaluate_population()
        logger.info("GA Optimization completed.")
        logger.info("Best final hyperparameters:")
        logger.info(self.best_chromosome.hyperparams)
    
    def evaluate_population(self):
        """
        Evaluate each chromosome in the population using the model training.
        """
        # Prepare arguments for evaluation
        positions = [chrom.hyperparams for chrom in self.population]
        seeds = [self.seed + i for i in range(len(self.population))]
        
        if self.use_multiprocessing:
            with multiprocessing.Pool(processes=min(self.population_size, multiprocessing.cpu_count())) as pool:
                fitnesses = pool.starmap(evaluate_wrapper, zip(positions, seeds))
        else:
            fitnesses = []
            for i, (pos, s) in enumerate(zip(positions, seeds)):
                logger.info(f"Evaluating chromosome {i+1}/{len(positions)}")
                score = evaluate_wrapper(pos, s)
                fitnesses.append(score)
        
        # Update fitness and global best
        for chrom, fit in zip(self.population, fitnesses):
            chrom.fitness = fit
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_chromosome = chrom.copy()
    
    def tournament_selection(self) -> Chromosome:
        """
        Tournament selection: pick a few chromosomes at random, 
        return the best among them.
        """
        contenders = random.sample(self.population, self.tournament_size)
        best = min(contenders, key=lambda c: c.fitness)
        return best.copy()
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Uniform crossover: for each param, pick parent1 or parent2’s gene 
        with probability p_crossover / 2.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        if random.random() < self.p_crossover:
            for param in self.search_space:
                if param == 'step':
                    continue
                if random.random() < 0.5:
                    # swap
                    child1.hyperparams[param], child2.hyperparams[param] = child2.hyperparams[param], child1.hyperparams[param]
        return child1, child2
    
    def mutate(self, chromosome: Chromosome):
        """
        Randomly mutate each gene with probability p_mutation. 
        """
        for param, bounds in self.search_space.items():
            if param == 'step':
                continue
            if random.random() < self.p_mutation:
                min_val, max_val = bounds
                if isinstance(min_val, int):
                    step = self.search_space.get('step', {}).get(param, 1)
                    possible_values = list(range(min_val, max_val + 1, step))
                    chromosome.hyperparams[param] = random.choice(possible_values)
                else:
                    # mutate in continuous range
                    chromosome.hyperparams[param] = random.uniform(min_val, max_val)


def install_ninja():
    """
    Ensure that the 'ninja' package is installed.
    """
    try:
        import ninja  # noqa
    except ModuleNotFoundError:
        install_package('ninja')


def main():
    """
    Main function to parse arguments and run the GA optimizer.
    """
    global logger
    
    install_ninja()
    
    parser = argparse.ArgumentParser("GA-GAN for DDGAN")
    
    parser.add_argument('--search_space', type=str, default='./configs/search_space_params.json',
                        help='Path to JSON file for search-space parameters')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to JSON file for DDGAN configuration')
    parser.add_argument('--save_dir', type=str, default='./converted_images',
                        help='Path to save images generated during FID score computation')
    parser.add_argument('--num_particles', type=int, default=10,   # parallels population_size
                        help='Number of chromosomes in the GA population')
    parser.add_argument('--num_iterations', type=int, default=20,  # parallels num_generations
                        help='Number of generations to perform')
    parser.add_argument('--limited_iteration_mode', type=int, default=202,
                        help='Limited iteration mode parameter')
    parser.add_argument('--with_FID', action='store_true', help='Compute FID score during evaluation')
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
    parser.add_argument('--use_multiprocessing', action='store_true', help='Use multiprocessing during evaluation')
    parser.add_argument('--log_file', type=str, default='ga_gan_optimization.log',
                        help='Path to the log file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Initial batch size to use')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    
    # GA hyperparams
    parser.add_argument('--p_crossover', type=float, default=0.8, help='Probability of crossover')
    parser.add_argument('--p_mutation', type=float, default=0.1, help='Probability of mutation')
    parser.add_argument('--tournament_size', type=int, default=2, help='Tournament size for selection')
    
    args = parser.parse_args()
    
    if args.log_file != 'ga_gan_optimization.log':
        logger.handlers.clear()
        logger = setup_logger(log_file=args.log_file)
    
    if args.use_multiprocessing:
        logger.info("Starting with multiprocessing")
        multiprocessing.set_start_method('spawn', force=True)
    
    set_random_seeds(args.seed)
    
    # Load config
    if args.config_file and os.path.isfile(args.config_file):
        config = load_json_to_dict(args.config_file, local=True)
        save_dict_to_json(config, filename='./configs/config.json', local=True)
        logger.info(f"Config file loaded from: {args.config_file}")
    else:
        logger.info("No config file provided. Using default configuration.")
        if not os.path.isfile('./configs/config.json'):
            run_bash_command(f"{find_python_command()} {os.curdir}/additionals/create_conf_default.py")
        
        config = load_json_to_dict('./configs/config.json', local=True)
    
    # Modify the baseline config to reflect command-line arguments
    to_add = {
        'save_dir': args.save_dir,
        'limited_iter': args.limited_iteration_mode,
        'resume': args.resume,
        'distributed': False,
        'batch_size': args.batch_size,
        'num_workers': 0,
        'with_FID': args.with_FID,
        'seed': args.seed
    }
    modify_json_file('./configs/config.json', to_add, local=True)
    
    # Load search space
    with open(args.search_space, 'r') as f:
        search_space = json.load(f)
    
    # We remove batch_size from search_space if present, 
    # because it’s being set from command line
    search_space.pop('batch_size', None)
    if 'step' in search_space:
        search_space['step'].pop('batch_size', None)
    
    # Initialize GA
    ga = GeneticAlgorithm(
        search_space=search_space,
        population_size=args.num_particles,
        num_generations=args.num_iterations,
        p_crossover=args.p_crossover,
        p_mutation=args.p_mutation,
        tournament_size=args.tournament_size,
        use_multiprocessing=args.use_multiprocessing,
        seed=args.seed
    )
    
    # Run GA
    ga.optimize()
    
    # Save the best hyperparameters
    if ga.best_chromosome is not None:
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(ga.best_chromosome.hyperparams, f, indent=4)
    logger.info("Genetic Algorithm optimization completed.")
    if ga.best_chromosome is not None:
        logger.info("Best hyperparameters found:")
        logger.info(json.dumps(ga.best_chromosome.hyperparams, indent=4))
    else:
        logger.info("No best chromosome found (something went wrong).")


if __name__ == '__main__':
    main()


#cloner174
