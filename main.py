from final_script_2 import main

class Args:
    pass

args = Args()
args.seed = 1024                # seed used for initialization
args.resume = False             # whether to resume checkpoints or start over
args.num_workers = 2            # Number of workers for Data Loader

args.mode = 'train'             #help='The mode of experience', choices=['train', 'test', 'val']
args.disc_small = 'yes'         #help='Use Small Discriminator?', choices=[ 'yes', 'no']
args.what_backend = 'nccl'      #choices=['nccl', 'gloo'], help='backend to use inside init_process_group'


args.do_resize = 'no'           #   choices=['yes', 'no'], ! transormers Setting !'
args.use_normalize = 'no'       #   choices=['yes', 'no'], ! transormers Setting !'
args.CenterCrop = 'no'          #   choices=['yes', 'no'], ! transormers Setting !'
args.centered = True            # bool:   help='-1,1 scale'


args.dataset = 'posluna'        # choices=['custom', 'posluna']  # help='name of dataset'




args.data_dir = '/content/Drive/MyDrive/cloner174/Luna16/data/Slices'

args.exp = 'exp1'
args.image_size = 64
args.num_channels = 3
args.num_channels_dae = 128             #help='number of initial channels in denosing model'
args.num_timesteps = 1
args.num_res_blocks = 2                 # help='number of resnet blocks per scale'

args.batch_size = 16
args.num_epoch = 2

args.ngf = 64
args.nz = 100
args.z_emb_dim = 256
args.t_emb_dim = 256
args.n_mlp = 4                          # help='number of mlp layers for z'
args.embedding_type = 'positional'      # choices=['positional', 'fourier']  # help='type of time embedding'

args.use_ema = True                     # help='use EMA or not'
args.ema_decay = 0.9999

args.r1_gamma = 0.02                    # help='coef for r1 reg'
args.lr_d = 1.25e-4                     # help='learning rate d'
args.lr_g = 1.6e-4                      # help='learning rate g'
args.lazy_reg = 15                      # help='lazy regulariation.'

args.beta_min = 0.1             # help='beta_max for diffusion'
args.beta_max = 20.0            # help='beta_min for diffusion'

args.beta1 = 0.5                        # help='beta1 for adam'
args.beta2 = 0.9                        # help='beta2 for adam'
args.no_lr_decay = False

args.progressive = 'none'               # choices=['none', 'output_skip', 'residual']    # help='progressive type for output'
args.progressive_input = 'residual'     # choices=['none', 'input_skip', 'residual']    # help='progressive type for input'
args.progressive_combine = 'sum'        # choices=['sum', 'cat']   # help='progressive combine method.'

args.attn_resolutions = (16,)           # help='resolution of applying attention'

args.dropout = 0.0                      # help='drop-out rate'

args.resamp_with_conv = True            # help='always up/down sampling with conv'
args.conditional = True                 #help='noise conditional'

args.fir = True
args.fir_kernel = [1, 3, 3, 1]

args.skip_rescale = True
args.resblock_type = 'biggan'           #  help='tyle of resnet block, choice in biggan and ddpm'
args.fourier_scale = 16.0               # help='scale of fourier transform'
args.not_use_tanh = False
args.use_geometric = False




args.num_process_per_node = 1           # help='number of gpus'
args.ch_mult = [1, 2, 2, 2]

args.num_proc_node = 1                  # help='The number of nodes in multi node env.'
args.node_rank = 0                      # help='The index of node.'
args.local_rank = 0                     # help='rank of process in the node'
args.master_address = '127.0.0.1'


args.save_content = True
args.save_content_every = 2
args.save_ckpt_every = 2

main(args)