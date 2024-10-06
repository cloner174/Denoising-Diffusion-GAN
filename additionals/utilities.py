#
import os
import sys
import json
import shutil
import subprocess



def copy_file(source_path, destination_path, replace=True, rename=None):
    """
    Copy a file from source_path to destination_path.
    replace: If True, replace the existing file if it exists.
    rename: If provided, rename the copied file to the specified name.
    """
    if os.path.isfile(source_path):
        final_destination = destination_path
        if rename:
            final_destination = os.path.join(os.path.dirname(destination_path), rename)
        
        elif os.path.exists(destination_path) and not replace:
            print(f"File already exists at {destination_path}. Set replace=True or provide a new name with rename parameter.")
            return
        
        shutil.copy2(source_path, final_destination)
        print(f"File copied from {source_path} to {final_destination}")
    
    else:
        print(f"Source file does not exist: {source_path}")


def copy_directory(source_path, destination_path, replace=True, rename=None):
    """
    Copy a directory from source_path to destination_path.
    replace: If True, replace the existing directory if it exists.
    rename: If provided, rename the copied directory to the specified name.
    """
    if os.path.isdir(source_path):
        
        final_destination = destination_path
        if rename:
            final_destination = os.path.join(os.path.dirname(destination_path), rename)
        
        elif os.path.exists(destination_path) and not replace:
            print(f"Directory already exists at {destination_path}. Set replace=True or provide a new name with rename parameter.")
            return
        
        if os.path.exists(final_destination) and replace:
            shutil.rmtree(final_destination)
        
        shutil.copytree(source_path, final_destination)
        print(f"Directory copied from {source_path} to {final_destination}")
    
    else:
        print(f"Source directory does not exist: {source_path}")


def move_file(source_path, destination_path, replace=True, rename=None):
    """
    Move a file from source_path to destination_path.
    replace: If True, replace the existing file if it exists.
    rename: If provided, rename the moved file to the specified name.
    """
    if os.path.isfile(source_path):
        
        final_destination = destination_path
        if rename:
            final_destination = os.path.join(os.path.dirname(destination_path), rename)
        
        elif os.path.exists(destination_path) and not replace:
            print(f"File already exists at {destination_path}. Set replace=True or provide a new name with rename parameter.")
            return
        
        if os.path.exists(final_destination) and replace:
            os.remove(final_destination)
        
        shutil.move(source_path, final_destination)
        print(f"File moved from {source_path} to {final_destination}")
    
    else:
        print(f"Source file does not exist: {source_path}")


def move_directory(source_path, destination_path, replace=True, rename=None):
    """
    Move a directory from source_path to destination_path.
    replace: If True, replace the existing directory if it exists.
    rename: If provided, rename the moved directory to the specified name.
    """
    if os.path.isdir(source_path):
        
        final_destination = destination_path
        if rename:
            final_destination = os.path.join(os.path.dirname(destination_path), rename)
        
        elif os.path.exists(destination_path) and not replace:
            print(f"Directory already exists at {destination_path}. Set replace=True or provide a new name with rename parameter.")
            return
        
        if os.path.exists(final_destination) and replace:
            shutil.rmtree(final_destination)
        
        shutil.move(source_path, final_destination)
        print(f"Directory moved from {source_path} to {final_destination}")
    
    else:
        print(f"Source directory does not exist: {source_path}")


def run_bash_command(command):
    """
    Run a bash command inside Python (e.g., in Google Colab).
    """
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")



def save_dict_to_json(data, filename, local = False):
    """
    Save a dictionary to a JSON file.
    :param data: Dictionary to save
    :param filename: Name of the JSON file
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    if not local:
        print(f"Saved config file to: {filename}")


def load_json_to_dict(filename, local= False):
    """
    Load data from a JSON file to a dictionary.
    :param filename: Name of the JSON file
    :return: Loaded dictionary
    """
    with open(filename, 'r') as json_file:
        d = json.load(json_file)
    
    if local:
        return d
    else:
        print(f"Config file {filename} has been loaded Successfully!")
        return d


def modify_json_file(filename, modifications):
    """
    Load, modify, and save changes to a JSON file.
    :param filename: Name of the JSON file
    :param modifications: Dictionary with modifications to apply
    """
    data = load_json_to_dict(filename, local = True)
    data.update(modifications)
    save_dict_to_json(data, filename, local=True)
    print(f"Config file {filename} Updated !")
    print("Changes: ", list(modifications.keys()))


def find_python_command():
    python_path = sys.executable
    python_command = os.path.basename(python_path)
    
    return python_command

def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print(f"Package '{package_name}' installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install package '{package_name}'. Error: {e}")
        print(f"Try runnig -> pip install {package_name} <-, manually")


#copy_file('source.txt', 'destination.txt', replace=False, rename='new_name.txt')
#copy_directory('source_folder', 'destination_folder', replace=False, rename='new_folder')
#move_file('source.txt', 'destination_folder/destination.txt', replace=True, rename='moved_name.txt')
#move_directory('source_folder', 'destination_folder', replace=True, rename='moved_folder')
#run_bash_command('ls -la')