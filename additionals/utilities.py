#
import os
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


#copy_file('source.txt', 'destination.txt', replace=False, rename='new_name.txt')
#copy_directory('source_folder', 'destination_folder', replace=False, rename='new_folder')
#move_file('source.txt', 'destination_folder/destination.txt', replace=True, rename='moved_name.txt')
#move_directory('source_folder', 'destination_folder', replace=True, rename='moved_folder')
#run_bash_command('ls -la')