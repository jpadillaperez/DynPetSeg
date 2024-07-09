import os
import nibabel as nib
import numpy as np

def combine_vessels(input_folders):
    for folder in input_folders:
        # Define paths for the left and right kidney files
        aorta = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'aorta.nii.gz')
        brachiocephalic_trunk = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'brachiocephalic_trunk.nii.gz')
        brachiocephalic_vein_left = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'brachiocephalic_vein_left.nii.gz')
        brachiocephalic_vein_right = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'brachiocephalic_vein_right.nii.gz')
        common_carotid_artery_left = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'common_carotid_artery_left.nii.gz')
        common_carotid_artery_right = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'common_carotid_artery_right.nii.gz')
        #heart_atrium_left = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'heart_atrium_left.nii.gz')
        #heart_atrium_right = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'heart_atrium_right.nii.gz')
        pulmonary_artery = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'pulmunary_artery.nii.gz')
        pulmonary_vein = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'pulmunary_vein.nii.gz')
        subclavian_artery_left = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'subclavian_artery_left.nii.gz')
        subclavian_artery_right = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'subclavian_artery_right.nii.gz')
        superior_vena_cava = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentationAorta', 'superior_vena_cava.nii.gz')
        
        heart_atrium_left = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentation', 'heart_atrium_left.nii.gz')
        heart_atrium_right = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentation', 'heart_atrium_right.nii.gz')
        heart_ventricle_left = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentation', 'heart_ventricle_left.nii.gz')
        heart_ventricle_right = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentation', 'heart_ventricle_right.nii.gz')

        output_folder = os.path.join(folder, 'NIFTY', 'Resampled', 'segmentation_processed')
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
    
        if os.path.exists(aorta) and os.path.exists(brachiocephalic_trunk) and os.path.exists(brachiocephalic_vein_left) and os.path.exists(brachiocephalic_vein_right) and os.path.exists(common_carotid_artery_left) and os.path.exists(common_carotid_artery_right) and os.path.exists(heart_atrium_left) and os.path.exists(heart_atrium_right) and os.path.exists(pulmonary_artery) and os.path.exists(pulmonary_vein) and os.path.exists(subclavian_artery_left) and os.path.exists(subclavian_artery_right) and os.path.exists(superior_vena_cava):
            # Load the NIfTI files
            aorta_img = nib.load(aorta)
            brachiocephalic_trunk_img = nib.load(brachiocephalic_trunk)
            brachiocephalic_vein_left_img = nib.load(brachiocephalic_vein_left)
            brachiocephalic_vein_right_img = nib.load(brachiocephalic_vein_right)
            common_carotid_artery_left_img = nib.load(common_carotid_artery_left)
            common_carotid_artery_right_img = nib.load(common_carotid_artery_right)
            heart_atrium_left_img = nib.load(heart_atrium_left)
            heart_atrium_right_img = nib.load(heart_atrium_right)
            heart_ventricle_left_img = nib.load(heart_ventricle_left)
            heart_ventricle_right_img = nib.load(heart_ventricle_right)
            pulmonary_artery_img = nib.load(pulmonary_artery)
            pulmonary_vein_img = nib.load(pulmonary_vein)
            subclavian_artery_left_img = nib.load(subclavian_artery_left)
            subclavian_artery_right_img = nib.load(subclavian_artery_right)
            superior_vena_cava_img = nib.load(superior_vena_cava)
            
            # Get the data from the NIfTI files
            aorta_data = aorta_img.get_fdata()
            brachiocephalic_trunk_data = brachiocephalic_trunk_img.get_fdata()
            brachiocephalic_vein_left_data = brachiocephalic_vein_left_img.get_fdata()
            brachiocephalic_vein_right_data = brachiocephalic_vein_right_img.get_fdata()
            common_carotid_artery_left_data = common_carotid_artery_left_img.get_fdata()
            common_carotid_artery_right_data = common_carotid_artery_right_img.get_fdata()
            heart_atrium_left_data = heart_atrium_left_img.get_fdata()
            heart_atrium_right_data = heart_atrium_right_img.get_fdata()
            heart_ventricle_left_data = heart_ventricle_left_img.get_fdata()
            heart_ventricle_right_data = heart_ventricle_right_img.get_fdata()
            pulmonary_artery_data = pulmonary_artery_img.get_fdata()
            pulmonary_vein_data = pulmonary_vein_img.get_fdata()
            subclavian_artery_left_data = subclavian_artery_left_img.get_fdata()
            subclavian_artery_right_data = subclavian_artery_right_img.get_fdata()
            superior_vena_cava_data = superior_vena_cava_img.get_fdata()
            
            # Create a combined data array, initializing with zeros
            combined_data = np.zeros(aorta_data.shape)
            
            # Set left and right kidney to 1
            combined_data[aorta_data > 0] = 1
            combined_data[brachiocephalic_trunk_data > 0] = 1
            combined_data[brachiocephalic_vein_left_data > 0] = 1
            combined_data[brachiocephalic_vein_right_data > 0] = 1
            combined_data[common_carotid_artery_left_data > 0] = 1
            combined_data[common_carotid_artery_right_data > 0] = 1
            combined_data[heart_atrium_left_data > 0] = 1
            combined_data[heart_atrium_right_data > 0] = 1
            combined_data[heart_ventricle_left_data > 0] = 1
            combined_data[heart_ventricle_right_data > 0] = 1
            combined_data[pulmonary_artery_data > 0] = 1
            combined_data[pulmonary_vein_data > 0] = 1
            combined_data[subclavian_artery_left_data > 0] = 1
            combined_data[subclavian_artery_right_data > 0] = 1
            combined_data[superior_vena_cava_data > 0] = 1
            
            # Create a new NIfTI image
            combined_img = nib.Nifti1Image(combined_data, aorta_img.affine, aorta_img.header)
            
            # Define the output file path
            output_file_path = os.path.join(output_folder, 'vessels.nii.gz')
            
            # Save the combined image
            nib.save(combined_img, output_file_path)
            
            print(f'Successfully combined vessels for {folder} and saved to {output_file_path}')
        else:
            print(f'Missing vessel files in folder: {folder}')

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

# Combine vessels and save
combine_vessels(input_folders)