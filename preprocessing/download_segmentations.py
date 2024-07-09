import os
import paramiko
from scp import SCPClient
    

input_folders = [
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/01_DynamicFDG_01/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/02_DynamicFDG_02/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/03_DynamicFDG_03/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/04_DynamicFDG_06/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/05_DynamicFDG_07/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/06_DynamicFDG_08/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/07_DynamicFDG_09/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/08_DynamicFDG_10/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/10_DynamicFDG_14/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/11_DynamicFDG_15/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/12_DynamicFDG_16/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/13_DynamicFDG_17/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/14_DynamicFDG_18/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/15_DynamicFDG_19/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/16_DynamicFDG_20/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/17_DynamicFDG_21/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/18_DynamicFDG_23/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/19_DynamicFDG_24/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/21_DynamicFDG_26/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/22_DynamicFDG_27/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/23_DynamicFDG_28/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/24_DynamicFDG_29/NIFTY/Resampled/segmentation',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/01_DynamicFDG_01/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/02_DynamicFDG_02/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/03_DynamicFDG_03/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/04_DynamicFDG_06/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/05_DynamicFDG_07/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/06_DynamicFDG_08/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/07_DynamicFDG_09/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/08_DynamicFDG_10/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/10_DynamicFDG_14/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/11_DynamicFDG_15/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/12_DynamicFDG_16/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/13_DynamicFDG_17/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/14_DynamicFDG_18/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/15_DynamicFDG_19/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/16_DynamicFDG_20/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/17_DynamicFDG_21/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/18_DynamicFDG_23/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/19_DynamicFDG_24/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/21_DynamicFDG_26/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/22_DynamicFDG_27/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/23_DynamicFDG_28/NIFTY/Resampled/segmentationAorta',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/24_DynamicFDG_29/NIFTY/Resampled/segmentationAorta'
]


import os
import paramiko
from scp import SCPClient

def create_ssh_client(server, port, user, password):
    """
    Create an SSH client.
    """
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def list_remote_files(ssh, remote_folder, pattern="*.nii.gz"):
    """
    List files in the remote folder matching the pattern.
    """
    stdin, stdout, stderr = ssh.exec_command(f"ls {remote_folder}/{pattern}")
    files = stdout.read().decode().split()
    return files

def download_data(input_folders, remote_user, remote_host, remote_password, local_dir):
    """
    Download NIfTI files from remote directories to a local directory, preserving the folder structure.

    Parameters:
    input_folders (list of str): List of remote directories containing NIfTI files.
    remote_user (str): Username for the remote server.
    remote_host (str): Hostname or IP address of the remote server.
    remote_password (str): Password for the remote server.
    local_dir (str): Local directory where files should be saved.
    """
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    ssh = create_ssh_client(remote_host, 22, remote_user, remote_password)
    scp = SCPClient(ssh.get_transport())

    for folder in input_folders:
        print(f"Listing files in {folder}...")
        remote_files = list_remote_files(ssh, folder)
        if not remote_files:
            print(f"No .nii.gz files found in {folder}")
            continue

        # Create the local directory structure
        relative_path = os.path.relpath(folder, '/home/guests/jorge_padilla/data/DynamicPET_Segmentation')
        local_path = os.path.join(local_dir, relative_path)
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        print(f"Created local directory: {local_path}")

        for remote_file in remote_files:
            remote_file_path = os.path.join(folder, os.path.basename(remote_file))
            print(f"Downloading {remote_file_path} to {local_path}...")
            try:
                scp.get(remote_file_path, local_path)
                print(f"Downloaded {remote_file_path} to {local_path}")
            except Exception as e:
                print(f"Failed to download {remote_file_path}: {e}")

    scp.close()
    ssh.close()

remote_user = 'jorge_padilla'
remote_host = '100.99.134.56'
remote_password = '16347157'  # Replace with your SSH password
local_dir = r'C:\Users\Jorge\Desktop\debug_data'

download_data(input_folders, remote_user, remote_host, remote_password, local_dir)