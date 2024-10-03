from PSO import PSO
import argparse
import json

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--search_space', type=str, default= './search_space_params.json', help= 'Path to Json File for Search-Space Params')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_particles', type=int, default=10)
    parser.add_argument('--num_iterations', type=int, default=20)
    
    args = parser.parse_args()
    
    
    with open(args.search_space, 'r') as f:
        search_space = json.load(f)
    
    search_space['step']['batch_size'] = args.batch_size
    
    pso = PSO(search_space, args.num_particles, args.num_iterations)
    
    pso.optimize()

#cloner174