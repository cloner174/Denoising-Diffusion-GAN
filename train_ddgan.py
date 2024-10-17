import argparse
from ddgan import main
import os
from additionals.utilities import run_bash_command, find_python_command, install_package, \
    load_json_to_dict, save_dict_to_json, modify_json_file


if __name__ == '__main__':
    
    config_dir = './configs'
    config_name = 'config.json'
    
    try:
        import ninja
    except ModuleNotFoundError:
        try:
            run_bash_command("pip install ninja")
        except:
            try:
                run_bash_command(f"{find_python_command()} -m pip install ninja ")
            except:
                install_package('ninja')
    
    
    parser = argparse.ArgumentParser('ddgan for Luna16')
    
    parser.add_argument('--use_config_file', default=True, 
                    help='If True, the arguments from the command line will be ignored, and the default configuration file will be used.')
    
    parser.add_argument('--config_file', default=None, 
                    help='Path to the configuration file to use. If provided, the arguments from the command line will be ignored. If an error occurs, no defaults will be used, and an exception will be raised.')
    
    parser.add_argument('--limited_slices', default = False, 
                        help='Whether to use all slices form npy files, or a part of them ?')
    
    parser.add_argument('--data_dir', default='./data', help='path to image files')
    parser.add_argument('--mask_dir', type=str, default=None, help='path to masks for images')
    
    parser.add_argument('--to_tensor_transform', type=str, default='yes', help='should to_tensor transform apply?')
    
    parser.add_argument('--bound_expand_limit', type=int, default=5, help='How many indexes should expand the bounds of images!!')
    
    parser.add_argument('--dataset', type=str, default='luna16', help='name of dataset', choices=['custom','posluna', 'luna16'])
    
    parser.add_argument('--resume', action='store_true',default=False)
    
    parser.add_argument('--seed', type=int, default=1024,help='seed used for initialization')
    
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for Data Loader')
    
    parser.add_argument('--mode', type=str, default='train', help='The mode of experience', choices=['train', 'test', 'val'])
    
    parser.add_argument('--disc_small', type=str, default='yes', help='Use Small Discriminator?', choices=[ 'yes', 'no'])
    
    
    # For distributed training
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training"
    )

    # For gradient clipping
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping"
    )

    # For weight decay
    parser.add_argument(
        "--weight_decay_G",
        type=float,
        default=0.0,
        help="Weight decay for Generator optimizer"
    )
    parser.add_argument(
        "--weight_decay_D",
        type=float,
        default=0.0,
        help="Weight decay for Discriminator optimizer"
    )

    # Separate beta parameters
    parser.add_argument("--beta1_g", type=float, default=0.5, help="Beta1 for Generator optimizer")
    parser.add_argument("--beta2_g", type=float, default=0.999, help="Beta2 for Generator optimizer")
    parser.add_argument("--beta1_d", type=float, default=0.5, help="Beta1 for Discriminator optimizer")
    parser.add_argument("--beta2_d", type=float, default=0.999, help="Beta2 for Discriminator optimizer")

    # For multiple discriminator updates
    parser.add_argument(
        "--d_updates_per_g_update",
        type=int,
        default=1,
        help="Number of Discriminator updates per Generator update"
    )

    
    parser.add_argument('--what_backend', default='nccl',choices=['nccl', 'gloo'], help='backend to use inside init_process_group')
    
    parser.add_argument('--do_resize', default='no', choices=['yes', 'no'], help='what should be inside transormers!')
    parser.add_argument('--use_normalize', default='no', choices=['yes', 'no'], help='what should be inside transormers!')
    parser.add_argument('--CenterCrop', default='no', choices=['yes', 'no'] , help='what should be inside transormers!')
    
    parser.add_argument('--image_size', type=int, default=32,help='size of image')
    
    parser.add_argument('--num_channels', type=int, default=3,help='channel of image')
    
    parser.add_argument('--centered', action='store_false', default=True,help='-1,1 scale')
    
    parser.add_argument('--use_geometric', action='store_true',default=False)
    
    parser.add_argument('--beta_min', type=float, default= 0.1,help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,help='beta_max for diffusion')
    
    parser.add_argument('--num_channels_dae', type=int, default=128,help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),help='resolution of applying attention')
    
    parser.add_argument('--dropout', type=float, default=0.,help='drop-out rate')
    
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',help='tyle of resnet block, choice in biggan and ddpm')
    
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],help='progressive combine method.')
    
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],help='type of time embedding')
    
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    #geenrator and training
    parser.add_argument('--exp', default='experiment_luna_default', help='name of experiment')
    
    parser.add_argument('--nz', type=int, default=100)
    
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    
    parser.add_argument('--num_epoch', type=int, default=5)
    
    parser.add_argument('--ngf', type=int, default=64)
    
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,help='beta2 for adam')
    
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    
    parser.add_argument('--use_ema', action='store_true', default=False,help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,help='lazy regulariation.')
    
    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',help='address for master')
    
    args = parser.parse_args()
    if not args.use_config_file :
        pass
    else:
        config = None
        if args.config_file is not None:
            if os.path.isfile(args.config_file):
                try:
                    config = load_json_to_dict(args.config_file)
                    print(f"Config file is loaded from: {args.config_file}, And will be used!")
                except Exception as e:
                    config = None
                    print(f"There was an error during loading your config file: {args.config_file}, Error: {e}")
        
        if config is None and args.config_file is None:
            import warnings
            warnings.warn("This script will use the file 'create_conf_default.py' file to create a new default file to use all configuration defaults!")
            if not os.path.isfile(os.path.join(config_dir , config_name)):
                run_bash_command( f"{find_python_command()} {os.curdir}/additionals/create_conf_default.py" )
            
            config = load_json_to_dict(os.path.join(config_dir , config_name))
        
        args = argparse.Namespace(**config)
    
    main(args)

#cloner174