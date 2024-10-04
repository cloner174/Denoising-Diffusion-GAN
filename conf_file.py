#cloner174
#defults

config = {
    'seed': 1024,
    'resume': True,
    'num_workers': 2,
    
    'limited_iter': 'no',   # Choices: 'no', 500, [100,500,2000,3000]  # explain: , A Number: int , or a list of numbers! 
    
    'mode': 'train',  # Choices: 'train', 'test', 'val'
    'disc_small': 'yes',  # Choices: 'yes', 'no'
    
    'data_dir': './data',
    
    'what_backend': 'nccl',  # Choices: 'nccl', 'gloo'
    'do_resize': 'no',  # Choices: 'yes', 'no'
    'use_normalize': 'no',  # Choices: 'yes', 'no'
    'CenterCrop': 'no',  # Choices: 'yes', 'no'
    
    'image_size': 64,
    'num_channels': 1,
    
    'centered': True,  # Corresponds to --centered flag
    'use_geometric': False,  # Corresponds to --use_geometric flag
    
    'beta_min': 0.1,
    'beta_max': 20.0,
    'num_channels_dae': 128,
    'n_mlp': 4,
    'ch_mult': [1, 2, 2, 2],
    
    'num_res_blocks': 2,
    'attn_resolutions': (16,),
    
    'dropout': 0.0,
    
    'resamp_with_conv': True,
    'conditional': True,
    'fir': True,  # Corresponds to --fir flag
    'fir_kernel': [1, 3, 3, 1],
    'skip_rescale': True,
    'resblock_type': 'biggan',  # Choices: 'biggan', 'ddpm'
    'progressive': 'none',  # Choices: 'none', 'output_skip', 'residual'
    'progressive_input': 'residual',  # Choices: 'none', 'input_skip', 'residual'
    'progressive_combine': 'sum',  # Choices: 'sum', 'cat'
    
    'embedding_type': 'positional',  # Choices: 'positional', 'fourier'
    'fourier_scale': 16.0,
    'not_use_tanh': False,  # Corresponds to --not_use_tanh flag
    # Generator and Training
    
    'exp': 'exp1',
    'dataset': 'posluna',
    
    'nz': 100,
    'num_timesteps': 1,
    
    'z_emb_dim': 256,
    't_emb_dim': 256,
    
    'batch_size': 64,
    'num_epoch': 5,
    'ngf': 64,
    
    'lr_g': 0.00001,
    'lr_d': 0.0001,
    'beta1': 0.1,
    'beta2': 0.2,
    
    'no_lr_decay': False,  # Corresponds to --no_lr_decay flag
    'use_ema': False,  # Corresponds to --use_ema flag
    'ema_decay': 0.01,
    
    'r1_gamma': 0.2,
    'lazy_reg': 10,
    
    'save_content': True,  # Corresponds to --save_content flag
    'save_content_every': 1,
    'save_ckpt_every': 1,
    
    # Distributed Data Parallel (DDP)
    'num_proc_node': 1,
    'num_process_per_node': 1,
    'node_rank': 0,
    'local_rank': 0,
    'master_address': '127.0.0.1'
}