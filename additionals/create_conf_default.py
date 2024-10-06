import os
import argparse
from utilities import save_dict_to_json

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Craete Config File With Defaults;')
    
    parser.add_argument('--save_dir', default = './configs', help='directory where Config File to be saved!')
    parser.add_argument('--filename', default = 'config.json', help='filename for Config File')
    
    args = parser.parse_args()
    
    conf = {
        'seed': 1024,
        'limited_slices':True,
        'resume': False,
        'num_workers': 0,
        'limited_iter': 'no',
        'mode': 'train',
        'disc_small': 'yes',
        'data_dir': './all_ones_final',
        'what_backend': 'nccl',
        'do_resize': 'no',
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
        'dropout': 0.0,
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
        'exp': 'exp1',
        'dataset': 'posluna',
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