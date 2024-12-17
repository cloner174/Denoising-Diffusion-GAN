import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utilities import save_dict_to_json

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Craete Config File With Defaults;')
    
    parser.add_argument('--save_dir', default = './configs', help='directory where Config File to be saved!')
    parser.add_argument('--filename', default = 'config.json', help='filename for Config File')
    
    args = parser.parse_args()
    
    conf = {
        'seed': 1024,
        'kind_of_optim': 'adam',    #can be 'adam' or 'pso'
        "use_config_file": True,
        "config_file": "configs/config.json",
        "mask_dir": "None",
        "to_tensor_transform": "yes",
        "bound_expand_limit": 0,
        "axis_for_limit": "z",
        "use_3d_mode": False,
        "path_to_slices_info" : 'None',
        'limited_slices':False,
        'resume': False,
        'num_workers': 0,
        'limited_iter': 'no',
        'mode': 'train',
        'disc_small': 'yes',
        'data_dir': '.',
        
        'distributed': False, 
        'grad_clip_norm': 1.0, 
        'weight_decay_G': 0.0, 
        'weight_decay_D': 0.0, 
        'beta1_g': 0.5, 
        'beta2_g': 0.999, 
        'beta1_d': 0.5, 
        'beta2_d': 0.999, 
        'd_updates_per_g_update': 1,
        
        'what_backend': 'nccl',
        'do_resize': 'yes',
        'use_normalize': 'no',
        'CenterCrop': 'no',
        'image_size': 64,
        'num_channels': 1,
        'centered': True,
        'use_geometric': False,
        'beta_min': 0.1,
        'beta_max': 20.0,
        'num_channels_dae': 128,
        'n_mlp': 4,
        'num_res_blocks': 2,
        'attn_resolutions': (16,),
        'dropout': 0.05,
        'resamp_with_conv': True,
        'conditional': True,
        'fir': True,
        'skip_rescale': True,
        'resblock_type': 'biggan',
        'progressive': 'none',
        'progressive_input': 'residual',
        'progressive_combine': 'sum',
        'embedding_type': 'positional',
        'fourier_scale': 16.0,
        'not_use_tanh': False,
        'exp': 'dbt',
        'dataset': 'custom',
        'nz': 100,
        'num_timesteps': 1,
        'z_emb_dim': 256,
        't_emb_dim': 256,
        'batch_size': 16,
        'num_epoch': 2,
        'ngf': 64,
        'lr_g': 0.0003,
        'lr_d': 0.0002,
        'beta1': 0.0,
        'beta2': 0.9,
        'no_lr_decay': False,
        'use_ema': True,
        'ema_decay': 0.01,
        'r1_gamma': 10.0,
        'lazy_reg': 16,
        'save_content': True,
        'save_content_every': 1,
        'save_ckpt_every': 1,
        'ch_mult': [1, 2, 2, 2],
        'fir_kernel': [1, 3, 3, 1],
        'num_proc_node': 1,
        'num_process_per_node': 1,
        'node_rank': 0,
        'local_rank': 0,
        'master_address': '127.0.0.1'
    }
    if not os.path.isdir(args.save_dir):
        raise NotADirectoryError(f"{args.save_dir} is NOT a directory!")
    
    config_path = os.path.join(args.save_dir, args.filename) # './configs/config.json'
    
    save_dict_to_json(conf, config_path, local=True)
    
    print(f"A config file named '{args.filename}' with default parameters has been created and saved to: {config_path}")
    print("Now you can use the modify_json_file function from additionals/utilities.py to adjust the values!")

    
#cloner174
