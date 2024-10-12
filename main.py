import os
import argparse
from ddgan import main
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
    
    parser.add_argument('--config_file', default=None, help='path to config file to be used!')
    parser.add_argument('--data_dir', default = './all_ones_final', help='path to image files')
    
    parser.add_argument('--limited_slices', default = False, 
                        help='Whether to use all slices form npy files, or a part of them ?')
    
    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--exp', default='exp1', help='name of experiment')
    parser.add_argument('--dataset', default='posluna', help='name of dataset')
    
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--save_content', action='store_true',default=False)
    
    args = parser.parse_args()
    
    config = None
    if args.config_file is not None and os.path.isfile(args.config_file):
        try:
            config = load_json_to_dict(args.config_file)
            save_dict_to_json(config , filename = os.path.join( config_dir, config_name ), local=True)
        
        except Exception as e:
            print(f"There was an error during loading your config file: {args.config_file}, Error: {e}")
            import warnings
            warnings.warn("This script will use the file './configs/config.json' for all configuration defaults!")
    
    if config is None and args.config_file is None :
        if not os.path.isfile('./configs/config.json'):
            run_bash_command( f"{find_python_command()} {os.curdir}/additionals/create_conf_default.py" )
            run_bash_command( f"{find_python_command()} {os.curdir}/additionals/create_conf_default.py --save_dir {config_dir} --filename {config_name}" )
    
    config = load_json_to_dict(os.path.join( config_dir, config_name ) , local=True)
    
    modification = args.__dict__
    
    modify_json_file(os.path.join( config_dir, config_name ), args.__dict__)
    
    config = load_json_to_dict(os.path.join( config_dir, config_name ) , local=True)
    
    args = argparse.Namespace(**config)
    
    main(args)
    
#cloner174