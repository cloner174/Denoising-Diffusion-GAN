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
    
    parser.add_argument('--use_config_file',
                    help='If True, the arguments from the command line will be ignored, and the default configuration file will be used.')
    
    parser.add_argument('--config_file', 
                    help='Path to the configuration file to use. If provided, the arguments from the command line will be ignored. If an error occurs, no defaults will be used, and an exception will be raised.')
    
    parser.add_argument('--limited_slices',
                        help='Whether to use all slices form npy files, or a part of them ?')
    
    parser.add_argument('--data_dir', help='path to image files')
    parser.add_argument('--mask_dir', type=str, help='path to masks for images')
    
    parser.add_argument('--to_tensor_transform', type=str, help='should to_tensor transform apply?')
    
    parser.add_argument('--bound_expand_limit', type=int , help='How many indexes should expand the bounds of images!!')
    
    parser.add_argument('--dataset', type=str, default='luna16', help='name of dataset', choices=['custom','posluna', 'luna16'])
    
    parser.add_argument('--resume', action='store_true')
    
    parser.add_argument('--seed', type=int, help='seed used for initialization')
    
    parser.add_argument('--num_workers', type=int,help='Number of workers for Data Loader')
    
    parser.add_argument('--mode', type=str, help='The mode of experience', choices=['train', 'test', 'val'])
    
    parser.add_argument('--disc_small', type=str,  help='Use Small Discriminator?', choices=[ 'yes', 'no'])
    
    
    
    
    parser.add_argument('--what_backend', default='nccl',choices=['nccl', 'gloo'], help='backend to use inside init_process_group')
    
    parser.add_argument('--do_resize',choices=['yes', 'no'], help='what should be inside transormers!')
    parser.add_argument('--use_normalize', choices=['yes', 'no'], help='what should be inside transormers!')
    parser.add_argument('--CenterCrop', choices=['yes', 'no'] , help='what should be inside transormers!')
    
    parser.add_argument('--image_size', type=int, help='size of image')
    
    parser.add_argument('--num_channels', type=int, help='channel of image')
    
    parser.add_argument('--centered', action='store_false',help='-1,1 scale')
    
    parser.add_argument('--use_geometric', action='store_true')
    
    parser.add_argument('--beta_min', type=float, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float,help='beta_max for diffusion')
    
    parser.add_argument('--num_channels_dae', type=int, help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int,help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions',help='resolution of applying attention')
    
    parser.add_argument('--dropout', type=float, help='drop-out rate')
    
    parser.add_argument('--resamp_with_conv', action='store_false', help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', help='noise conditional')
    parser.add_argument('--fir', action='store_false', help='FIR')
    parser.add_argument('--fir_kernel', help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false',help='skip rescale')
    parser.add_argument('--resblock_type',help='tyle of resnet block, choice in biggan and ddpm')
    
    parser.add_argument('--progressive', type=str, choices=['none', 'output_skip', 'residual'],help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, choices=['none', 'input_skip', 'residual'],help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str,choices=['sum', 'cat'],help='progressive combine method.')
    
    parser.add_argument('--embedding_type', type=str, choices=['positional', 'fourier'],help='type of time embedding')
    
    parser.add_argument('--fourier_scale', type=float, help='scale of fourier transform')
    
    parser.add_argument('--not_use_tanh', action='store_true')
    #geenrator and training
    parser.add_argument('--exp', default='experiment_luna_default', help='name of experiment')
    
    parser.add_argument('--nz', type=int)
    
    parser.add_argument('--num_timesteps', type=int)
    
    parser.add_argument('--z_emb_dim', type=int)
    parser.add_argument('--t_emb_dim', type=int)
    
    parser.add_argument('--batch_size', type=int, help='input batch size')
    
    parser.add_argument('--num_epoch', type=int)
    
    parser.add_argument('--ngf', type=int, default=64)
    
    parser.add_argument('--lr_g', type=float,  help='learning rate g')
    parser.add_argument('--lr_d', type=float,  help='learning rate d')
    parser.add_argument('--beta1', type=float, help='beta1 for adam')
    parser.add_argument('--beta2', type=float,help='beta2 for adam')
    
    parser.add_argument('--no_lr_decay',action='store_true')
    
    parser.add_argument('--use_ema', action='store_true', help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, help='decay rate for EMA')
    
    parser.add_argument('--r1_gamma', type=float, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int,help='lazy regulariation.')
    
    parser.add_argument('--save_content', action='store_true')
    parser.add_argument('--save_content_every', type=int, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, help='save ckpt every x epochs')
    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',help='address for master')

    parser.add_argument('--fast_memory', default=False)
    
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
            up_ = args.__dict__
            ok_ = False
            while True:
              if ok_:
                break
              try:
                for akey_ , aval_ in up_.items():
                  if aval_ is None:
                    del up_[akey_]
                ok_ = True
              except:
                pass
            
            if len(up_) > 0 :
                modify_json_file(os.path.join(config_dir , config_name), up_)
                config = load_json_to_dict(os.path.join(config_dir , config_name))
        
        args = argparse.Namespace(**config)
    
    main(args)

#cloner174
