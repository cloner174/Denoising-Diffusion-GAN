from train_ddgan import main

class Args:
    pass

args = Args()
args.seed = 1024  # seed used for initialization
args.resume = False  # whether to resume checkpoints or start over
args.num_workers = 2  # Number of workers for Data Loader


args.data_dir = '/content/Drive/MyDrive/cloner174/Luna16/data/Slices'
args.dataset = 'custom'
args.exp = 'exp1'
args.image_size = 64
args.num_channels = 3
args.num_channels_dae = 128
args.num_timesteps = 1
args.num_res_blocks = 2
args.batch_size = 16
args.num_epoch = 2
args.ngf = 64
args.nz = 100
args.z_emb_dim = 256
args.n_mlp = 4
args.embedding_type = 'positional'
args.use_ema = True
args.ema_decay = 0.9999
args.r1_gamma = 0.02
args.lr_d = 1.25e-4
args.lr_g = 1.6e-4
args.lazy_reg = 15
args.num_process_per_node = 1
args.ch_mult = [1, 2, 2, 2]
args.save_content = True


args.num_proc_node = 1
args.node_rank = 0
args.local_rank = 0
args.master_address = '127.0.0.1'
args.beta1 = 0.5
args.beta2 = 0.9
args.no_lr_decay = False
args.progressive = 'none'
args.progressive_input = 'residual'
args.progressive_combine = 'sum'
args.attn_resolutions = (16,)
args.dropout = 0.0
args.resamp_with_conv = True
args.conditional = True
args.fir = True
args.fir_kernel = [1, 3, 3, 1]
args.skip_rescale = True
args.resblock_type = 'biggan'
args.fourier_scale = 16.0
args.not_use_tanh = False
args.use_geometric = False
args.beta_min = 0.1
args.beta_max = 20.0

main(args)