import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label, find_objects

def largest_connected_component(data):
    labeled_array, num_features = label(data)
    if num_features == 0:
        return data
    
    largest_component = None
    max_size = 0
    
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        component_size = component.sum()
        
        if component_size > max_size:
            max_size = component_size
            largest_component = component
            
    return largest_component.astype(np.uint8)

def combine_kidneys(input_folders):
    for folder in input_folders:
        # Define paths for the left and right kidney files
        left_kidney_path = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentation', 'kidney_left.nii.gz')
        right_kidney_path = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentation', 'kidney_right.nii.gz')

        output_folder = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentation_processed')
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        if os.path.exists(left_kidney_path) and os.path.exists(right_kidney_path):
            # Load the NIfTI files
            left_kidney_img = nib.load(left_kidney_path)
            right_kidney_img = nib.load(right_kidney_path)
            
            # Get the data from the NIfTI files
            left_kidney_data = left_kidney_img.get_fdata()
            right_kidney_data = right_kidney_img.get_fdata()
            
            # Keep only the largest connected component for each kidney
            left_kidney_largest_cc = largest_connected_component(left_kidney_data)
            right_kidney_largest_cc = largest_connected_component(right_kidney_data)
            
            # Create a combined data array, initializing with zeros
            combined_data = np.zeros(left_kidney_data.shape)
            
            # Set left and right kidney to 1
            combined_data[left_kidney_largest_cc > 0] = 1
            combined_data[right_kidney_largest_cc > 0] = 1
            
            # Create a new NIfTI image
            combined_img = nib.Nifti1Image(combined_data, left_kidney_img.affine, left_kidney_img.header)
            
            # Define the output file path
            output_file_path = os.path.join(output_folder, 'kidneys.nii.gz')
            
            # Save the combined image
            nib.save(combined_img, output_file_path)
            
            print(f'Successfully combined kidneys for {folder} and saved to {output_file_path}')
        else:
            print(f'Missing kidney files in folder: {folder}')

# List of input folders

input_folders = [
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/01_DynamicFDG_01',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/02_DynamicFDG_02',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/03_DynamicFDG_03',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/04_DynamicFDG_06',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/05_DynamicFDG_07',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/06_DynamicFDG_08',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/07_DynamicFDG_09',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/08_DynamicFDG_10',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/10_DynamicFDG_14',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/11_DynamicFDG_15',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/12_DynamicFDG_16',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/13_DynamicFDG_17',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/14_DynamicFDG_18',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/15_DynamicFDG_19',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/16_DynamicFDG_20',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/17_DynamicFDG_21',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/18_DynamicFDG_23',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/19_DynamicFDG_24',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/21_DynamicFDG_26',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/22_DynamicFDG_27',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/23_DynamicFDG_28',
    '/home/guests/jorge_padilla/data/DynamicPET_Segmentation/24_DynamicFDG_29'
]

# Combine kidneys and save
combine_kidneys(input_folders)
