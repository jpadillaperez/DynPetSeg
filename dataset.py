import os
import torch
import numpy as np
import glob
import SimpleITK as sitk
from monai.data import CacheDataset
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils_kinetic import PET_2TC_KM
from utils.utils_torch import torch_interp_1d
from utils.set_root_paths import root_data_path, root_dataset_path
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import label

class DynPETDataset(CacheDataset):
    def __init__(self, config, dataset_type, patch_size=None):
        # Enforce determinism
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # Read global config
        self.config = config
        self.dataset_type = dataset_type    #train, validation or test
        self.patch_size = patch_size

        print(f"Using patch size {self.patch_size} for {self.dataset_type}")

        # Create config for each dataset type from global config
        self.train_config = {"patient_list": self.config["patient_list"]["train"], "slices_per_patient": self.config["slices_per_patient"], "bounding_box": self.config["bounding_box"], "segmentation_list" : self.config["segmentation_list"]}
        self.val_config = {"patient_list": self.config["patient_list"]["validation"], "slices_per_patient": self.config["slices_per_patient"], "bounding_box": self.config["bounding_box"], "segmentation_list" : self.config["segmentation_list"]}
        self.test_config = {"patient_list": self.config["patient_list"]["test"], "slices_per_patient": self.config["slices_per_patient"], "bounding_box": self.config["bounding_box"], "segmentation_list" : self.config["segmentation_list"]}

        # Select the correct config
        self.idif = dict()

        self.data_list = list()
        self.data_seg_list = list()
        self.data_ct_list = list()

        if dataset_type == "train":
            self.build_dataset(self.train_config)
        elif dataset_type == "validation":
            self.build_dataset(self.val_config)
        elif dataset_type == "test":
            self.build_dataset(self.test_config)
        else: 
            print("ERROR: dataset type not supported!")
            return


    def __getitem__(self,idx):
        #DONE: load torch from here 
        item = torch.load(self.data_list[idx])
        item[2] = item[2].type(torch.cuda.FloatTensor)

        if ((self.config["experiment"] == "adaptation2") or (self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "segmentation")):
            #DONE: load segmentation from here
            item_seg = torch.load(self.data_seg_list[idx])[2].type(torch.cuda.IntTensor)
            item.append(item_seg)
            assert self.data_list[idx].split("_")[-1].split(".")[0] == self.data_seg_list[idx].split("_")[-1].split(".")[0], "Data and segmentation slices don't match"
            assert self.data_list[idx].split("_")[-2].split("data")[-1] == self.data_seg_list[idx].split("_")[-2].split("seg")[-1], "Data and segmentation patients don't match"
        
            #DONE: load CT from here
            item_ct = torch.load(self.data_ct_list[idx])[2].type(torch.cuda.FloatTensor)
            item.append(item_ct)
            assert self.data_list[idx].split("_")[-1].split(".")[0] == self.data_ct_list[idx].split("_")[-1].split(".")[0], "Data and CT slices don't match"
            assert self.data_list[idx].split("_")[-2].split("data")[-1] == self.data_ct_list[idx].split("_")[-2].split("ct")[-1], "Data and CT patients don't match"

        return item

    def __len__(self):
        return int(self.length)
    
    def __get_patch_size__(self):
        if self.patch_size == None:
            item = torch.load(self.data_list[0])
            item[2] = item[2].type(torch.cuda.FloatTensor)
            self.patch_size = item[2].shape[-1] 
        return self.patch_size
    
    def remove_slices_without_segmentation(self):
        data_list = list()
        data_seg_list = list()
        data_ct_list = list()
        for idx in range(len(self.data_list)):
            seg = torch.load(self.data_seg_list[idx])[2].type(torch.cuda.IntTensor)
            #Check if the segmentation has any value different from 0
            if torch.sum(seg) > 0:
                assert self.data_list[idx].split("_")[-1].split(".")[0] == self.data_seg_list[idx].split("_")[-1].split(".")[0], "Data and segmentation slices don't match"
                assert self.data_list[idx].split("_")[-2].split("data")[-1] == self.data_seg_list[idx].split("_")[-2].split("seg")[-1], "Data and segmentation patients don't match"
                
                data_list.append(self.data_list[idx])
                data_seg_list.append(self.data_seg_list[idx])
                data_ct_list.append(self.data_ct_list[idx])

        self.data_list = data_list
        self.data_seg_list = data_seg_list
        self.data_ct_list = data_ct_list

        self.length = len(self.data_list)

        print("New Dataset", self.dataset_type, "has", self.length, "slices and length", self.length, "\n")
        return
    
    def build_dataset(self, current_config):
        self.patient_list = current_config["patient_list"]
        self.slices_per_patient_mode = current_config["slices_per_patient"]
        self.slices_per_patient = dict()

        for p in self.patient_list:
            self.slices_per_patient[p] = dict()
            self.load_txt_data(p)

        #FIXME: calculate current_dataset_size properly with for loop over patients #FIXED
        #self.current_dataset_size = current_config["slices_per_patient"] * len(current_config["patient_list"])     
        current_size = 0
        for patient in self.patient_list:
            for pet in self.slices_per_patient[patient]:
                current_size += max(self.slices_per_patient[patient][pet]["Slice_Count"])
        self.current_dataset_size = current_size

        # Load existing data, if possible
        load_data = self.load_data(current_config) #FIXME: update load data
        if load_data is None:  
            # If the dataset does not exist, create the folder where it will be saved
            if not os.path.exists(self.save_data_folder):
                os.makedirs(self.save_data_folder)

            [self.data_list, self.data_seg_list, self.data_ct_list] = self.read_dynpet(current_config)
            print("Dataset", self.dataset_type, "was saved in", self.save_data_folder)
        else:                   
            [self.data_list, self.data_seg_list, self.data_ct_list] = load_data

        
        if self.config["overfit"]:
            print("Overfitting mode")
            self.data_list = [self.data_list[0]]
            self.data_seg_list = [self.data_seg_list[0]]
            self.data_ct_list = [self.data_ct_list[0]]

        
        assert len(self.data_list) == len(self.data_seg_list), "The number of data and segmentation files is different"
        assert len(self.data_list) == len(self.data_ct_list), "The number of data and CT files is different"
        
        self.length = len(self.data_list)

        print("Dataset", self.dataset_type, "has", self.current_dataset_size, "slices and length", self.length, "\n") #FIXME: This is fake, make it true for all slices #DONE

        return

    def load_txt_data(self, patient):
        tac_txt_path = os.path.join(root_data_path, "TAC", "DynamicFDG_"+patient+"_TAC.txt")
        idif_txt_path = os.path.join(root_data_path, "IDIF", "DynamicFDG_"+patient+"_IDIF.txt")
        info_txt_path = os.path.join(root_data_path, "INFO", "Resampled", "DynamicFDG_"+patient+"_INFO.txt") #TODO: READ FILE WITH SIZES FROM HERE #DONE

        # Read acquisition time
        data = pd.read_csv(tac_txt_path, sep="\t")
        data['start[seconds]'] = data['start[seconds]'].apply(lambda x: x/60)
        data['end[kBq/cc]'] = data['end[kBq/cc]'].apply(lambda x: x/60)
        data.rename(columns={"start[seconds]": "start[minutes]"}, inplace=True)
        data.rename(columns={"end[kBq/cc]": "end[minutes]"}, inplace=True)
        time_stamp = data['start[minutes]'].values
        self.time_stamp = torch.Tensor(np.around(time_stamp, 2))

        # Define interpolated time axis which is required to run the convolution
        step = 0.1
        self.t = torch.Tensor(np.arange(self.time_stamp[0], self.time_stamp[-1], step))

        # Read IDIF and interpolate it ì
        rolling_window = 1
        data = pd.read_csv(idif_txt_path, sep="\t").rolling(rolling_window).mean()
        idif_sample_time = torch.Tensor(data["sample-time[minutes]"])
        idif = torch.Tensor(data["plasma[kBq/cc]"])
        self.idif[patient] = torch_interp_1d(self.t, idif_sample_time, idif)

        #Save number of slices in dict
        data = pd.read_csv(info_txt_path, sep=",")

        for i in range(len(data)):
            if self.slices_per_patient_mode == "all":
                self.slices_per_patient[patient][data["PET"][i]] = {"Slice_Count": range(data["Slice_Count"][i]), "Slice_Size": data["Slice_Size"][i]}
            elif self.slices_per_patient_mode == "no_legs":
                self.slices_per_patient[patient][data["PET"][i]] = {"Slice_Count": range(int(data["Slice_Count"][i] / 2), data["Slice_Count"][i] - 1), "Slice_Size": data["Slice_Size"][i]}
            else:
                raise ValueError("slices_per_patient should be 'all' or 'no_legs'")
        return 

    def read_dynpet(self, current_config):
        print("Creating dataset", self.dataset_type, "with", self.current_dataset_size, "slices...")
        data_list = list()
        seg_list = list()
        ct_list = list()
        for patient in self.patient_list:
            print("\tProcessing patient: " + str(patient))
            patient_folder = glob.glob(os.path.join(root_data_path, "*DynamicFDG_"+patient))[0]
            # When using self.config["slices_per_patient_*"] --> probably the selection of self.slice can be shortened a little bit
            #self.slices_per_patients = int(self.current_dataset_size / len(self.patient_list)) #FIXME: Make a dictionary with patients and slices
            #DOUBT JPP: When using all of the slices per patient, it still means same number of slices in all the patients? --> No
            #DONE: Add a config flag to avoid this bbox step
            if current_config["bounding_box"]:
                # When config["slices_per_patient_*"]>1 (and therefore self.current_dataset_size > 1), the slices are selected within a ROI defined by a bouding box (bb). We used a bb
                # including the lungs and the bladder. In this way we didn't considered the head (because of the movement over the acquisition) and the legs (which are not very informative).
                # The use of the bb is not mandatory.  
                bb_path = patient_folder + "/NIFTY/Resampled/bb.nii.gz"
                bb_ = sitk.GetArrayFromImage(sitk.ReadImage(bb_path))

                # JPP: When using all of the slices per patient, we still need a bounding box? --> No, the bbox is just for other non segmentation purposes

                # First, the indexes of the slices are picked homogeneously withing the indexes of the bb
                indexes = np.nonzero(bb_)
                top = indexes[0][-1]
                bottom = indexes[0][0]
                step = np.floor((top - bottom) / self.slices_per_patient[patient])  #FIXME: slices should be different per patient
                if step == 0:       # This happens if the bb is smaller than the self.slices_per_patients (top - bottom < self.slices_per_patients)
                    step = 1
                hom_pick = torch.arange(bottom, top, step)

                # If the homogeneous pick is much bigger than the expected dataset size, the pick is reduced by lowering the sampling frequency at borders of bb.
                # The underlying assumption is that the most informative region is the center of the bb.
                # This step can be omitted (and just use pick = hom_pick)
                if len(hom_pick)-self.slices_per_patient[patient] > 50: #FIXME: No needed if we use slice per patient
                    center_slice = int(len(hom_pick)/2)
                    a = int((len(hom_pick)-self.slices_per_patient[patient]) * 2 / 3)
                    new_step = int(step+1)
                    hom_pick_short = torch.concat((hom_pick[:center_slice-a][::new_step], hom_pick[center_slice-a:center_slice+a], hom_pick[center_slice+a:-1][::new_step]))
                    if len(hom_pick_short) > self.slices_per_patient[patient]: pick = hom_pick_short
                    else: pick = hom_pick
                else: 
                    pick = hom_pick
 
                if top - bottom < self.slices_per_patient[patient]:            
                    # All the slices in the bb can be selected
                    self.slices = hom_pick[0:self.slices_per_patient[patient]]
                elif self.dataset_type == "test":
                    # When testing, we always use the homogeneous pick
                    self.slices = hom_pick[0:self.slices_per_patient[patient]] #FIXME: ugly trick, use for test the same as for train 
                else:
                    # In all the other cases, self.slices_per_patients are selected in the center of the bb
                    c = int(len(pick)/2)
                    s = int(self.slices_per_patient[patient]/2)
                    self.slices = pick[c-s:c+s+1]                       # Select N=self.slices_per_patients slices in mid of pick
                
                # len(self.slices) may not be exactly equal to self.slices_per_patients beacuse of numerical errors (their differece it's usually 1)
                print("\tN_slices=" + str(len(self.slices)) + "/" + str(top - bottom))

                #current_data contains the dynpet for just one patient
                current_data = torch.zeros((len(self.slices), len(pet_list), size, size))  #[SLICES, DYNPET, X, Y] TODO: better [DYNPET, SLICES, X, Y]?     
                
                #Iterate over all the pets of the dynamic pet of the same patients 
                for i in range(len(pet_list)):
                    print("\t\tPET: " + str(i) + "/" + str(len(pet_list)))

                    p = pet_list[i]
                    current_pet = torch.from_numpy(np.int32(sitk.GetArrayFromImage(sitk.ReadImage(p))))

                    # Define borders of center crop
                    slice_size = current_pet[0, :, :].shape
                    slice_center = torch.tensor(slice_size)[0] / 2

                    if size < slice_size[0] or size < slice_size[1]:
                        for j in range(len(self.slices)): 
                            slice = int(self.slices[j])
                            #Cut patch size from center of slice
                            current_slice = current_pet[slice, int(slice_center)-int(size/2):int(slice_center)+int(size/2), int(slice_center)-int(size/2):int(slice_center)+int(size/2)] 
                    else:
                        for j in range(len(self.slices)): 
                            slice = int(self.slices[j])
                            #Add padding with background value as minimum value of the pet
                            m, n = current_pet[slice, :, :].shape
                            padding = ((max(0, size - m), max(0, size - m)), (max(0, size - n), max(0, size - n)))
                            current_slice = np.pad(current_pet[slice, :, :], padding, 'constant', constant_values=(0, 0))
                            current_data[j, i, :, :] = current_slice/1000           # from Bq/ml to kBq/ml

            else:
                print("\t\tIgnoring bounding box..." )

                slices_in_patient = list()
                sizes_in_patient = list()
                pet_list = self.slices_per_patient[patient].keys()
                for pet in self.slices_per_patient[patient]:
                    sizes_in_patient.append(int(self.slices_per_patient[patient][pet]["Slice_Size"]))
                    slices_in_patient.append(self.slices_per_patient[patient][pet]["Slice_Count"])

                print("\t\tPets in patient: " + str(pet_list))
                print("\t\tSlices in patient: " + str(slices_in_patient))
                print("\t\tSizes in patient: " + str(sizes_in_patient))
                
                #Maximum size possible
                if self.patch_size == "max":
                    if self.patch_size != max(sizes_in_patient):
                        if max(sizes_in_patient) % 2 == 1:
                            print(f"WARNING: Setting patch size to {max(sizes_in_patient)-1}")
                            self.patch_size = max(sizes_in_patient)-1
                        else:
                            print(f"WARNING: Setting patch size to {max(sizes_in_patient)}")
                            self.patch_size = max(sizes_in_patient)

                size = self.patch_size
                max_num_slices = max(ss.stop for ss in slices_in_patient if ss)
                current_data = torch.zeros((max_num_slices, len(pet_list), size, size))  #[SLICES, DYNPET, X, Y] TODO: better [DYNPET, SLICES, X, Y]?  

                #Iterate over all the pets of the dynamic pet of the same patients with progress bar
                for i, p in enumerate(tqdm(pet_list, desc="Processing pets")):
                    current_pet = torch.from_numpy(np.int32(sitk.GetArrayFromImage(sitk.ReadImage(patient_folder + "/NIFTY/Resampled/PET_" + str(p) + ".nii.gz"))))
                    if size < current_pet[0, :, :].shape[0] or size < current_pet[0, :, :].shape[1]:
                        #If patch_size is smaller than the size of the original pet, make center crop
                        slice_center = torch.tensor(current_pet[0, :, :].shape)[0] / 2
                        for slice in slices_in_patient[i]:
                            current_slice = current_pet[slice, int(slice_center)-int(size/2):int(slice_center)+int(size/2), int(slice_center)-int(size/2):int(slice_center)+int(size/2)]
                            current_data[slice, i, :, :] = current_slice/1000           # from Bq/ml to kBq/ml
                    else:           
                        #If patch_size is bigger than the size of the original pet, make padding with background value as minimum value of the pet
                        for slice in slices_in_patient[i]:
                            m, n = current_pet[slice, :, :].shape
                            padding = ((max(0, size - m), max(0, size - m)), (max(0, size - n), max(0, size - n)))
                            current_slice = torch.from_numpy(np.pad(current_pet[slice, :, :], padding, 'constant', constant_values=(0, 0)))
                            current_data[slice, i, :, :] = current_slice/1000           # from Bq/ml to kBq/ml

                #----------------- GET SEGMENTATION DATA -----------------

                segmentation_list = list()
                if ("liver" in current_config["segmentation_list"]):
                    print("\t\tImporting liver from preprocessed segmentation...")
                    org_list = glob.glob(patient_folder + "/NIFTY/Resampled/segmentation_processed/liver.nii.gz")
                    segmentation_list += [s for s in org_list if s.split("/")[-1].split(".")[0] in "liver"]
                if ("kidneys" in current_config["segmentation_list"]): 
                    print("\t\tImporting kidneys from preprocessed segmentation...")
                    org_list = glob.glob(patient_folder + "/NIFTY/Resampled/segmentation_processed/kidneys.nii.gz")
                    segmentation_list += [s for s in org_list if s.split("/")[-1].split(".")[0] in "kidneys"]
                if ("vessels" in current_config["segmentation_list"]):
                    print("\t\tImporting vessels from preprocessed segmentation...")
                    org_list = glob.glob(patient_folder + "/NIFTY/Resampled/segmentation_processed/vessels.nii.gz")
                    segmentation_list += [s for s in org_list if s.split("/")[-1].split(".")[0] in "vessels"]
                if ("aorta" in current_config["segmentation_list"]):
                    print("\t\tImporting aorta from aorta segmentation...")
                    org_list = glob.glob(patient_folder + "/NIFTY/Resampled/segmentationAorta/aorta.nii.gz")
                    segmentation_list += [s for s in org_list if s.split("/")[-1].split(".")[0] in "aorta"]
                if segmentation_list == []:
                    print("\t\tERROR: Segmentation not found!")
                    return

                segmentation_list.sort()
                print("\t\tSegmentations in patient: " + str(segmentation_list))
                current_seg_data = torch.zeros((max_num_slices, len(segmentation_list) + 1, size, size))  #[SLICES, SEGMENTATION, X, Y] TODO: better [SEGMENTATION, SLICES, X, Y]?

                for i, s in enumerate(tqdm(segmentation_list, desc="Processing segmentations")):
                    current_segmentation = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(s)))

                    #Fixing the number of slices in the segmentation
                    if max_num_slices < current_segmentation.shape[0]:
                        current_segmentation = current_segmentation[int((current_segmentation.shape[0]-max_num_slices)/2):int((current_segmentation.shape[0]-max_num_slices)/2)+max_num_slices, :, :]
                    elif max_num_slices > current_segmentation.shape[0]:
                        padding = ((max(0, max_num_slices - current_segmentation.shape[0]), max(0, max_num_slices - current_segmentation.shape[0])), (0, 0), (0, 0))
                        current_segmentation = torch.from_numpy(np.pad(current_segmentation, padding, 'constant', constant_values=(0, 0)))

                    if size < current_segmentation[0, :, :].shape[0] or size < current_segmentation[0, :, :].shape[1]:
                        #If patch_size is smaller than the size of the original pet, make center crop
                        slice_center = torch.tensor(current_segmentation[0, :, :].shape)[0] / 2
                        for slice in np.array(slices_in_patient[0]).repeat(len(segmentation_list)):
                            current_slice = current_segmentation[slice, int(slice_center)-int(size/2):int(slice_center)+int(size/2), int(slice_center)-int(size/2):int(slice_center)+int(size/2)]
                            current_seg_data[slice, i + 1, :, :] = current_slice
                    else:
                        #If patch_size is bigger than the size of the original pet, make padding with background value as minimum value of the pet
                        for slice in np.array(slices_in_patient[0]).repeat(len(segmentation_list)):
                            m, n = current_segmentation[slice, :, :].shape
                            padding = ((max(0, size - m), max(0, size - m)), (max(0, size - n), max(0, size - n)))
                            current_slice = torch.from_numpy(np.pad(current_segmentation[slice, :, :], padding, 'constant', constant_values=(0, 0)))
                            current_seg_data[slice, i + 1, :, :] = current_slice

                assert current_data.shape[0] == current_seg_data.shape[0], "The number of slices in the PET and the segmentation are different"
                assert current_data.shape[2] == current_seg_data.shape[2], "The size of the PET and the segmentation are different"
                assert current_data.shape[3] == current_seg_data.shape[3], "The size of the PET and the segmentation are different"

                # Make the current segmentation data to be one single channel with hot end encoding on the dimension 1
                current_seg_data = np.argmax(current_seg_data, axis=1)
                current_seg_data = current_seg_data.squeeze(0)


                #----------------- GET CT DATA -----------------

                current_ct_data = torch.zeros((max_num_slices, 1, size, size))  #[SLICES, 1, X, Y] TODO: better [SEGMENTATION, SLICES, X, Y]?
                current_ct = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(patient_folder + "/NIFTY/Resampled/CT.nii.gz")))

                #Fixing the number of slices in the segmentation
                if max_num_slices < current_ct.shape[0]:
                    current_ct = current_ct[int((current_ct.shape[0]-max_num_slices)/2):int((current_ct.shape[0]-max_num_slices)/2)+max_num_slices, :, :]
                elif max_num_slices > current_ct.shape[0]:
                    padding = ((max(0, max_num_slices - current_ct.shape[0]), max(0, max_num_slices - current_ct.shape[0])), (0, 0), (0, 0))
                    current_ct = torch.from_numpy(np.pad(current_ct, padding, 'constant', constant_values=(0, 0)))

                if size < current_ct[0, :, :].shape[0] or size < current_ct[0, :, :].shape[1]:
                    #If patch_size is smaller than the size of the original pet, make center crop
                    slice_center = torch.tensor(current_ct[0, :, :].shape)[0] / 2
                    for slice in np.array(slices_in_patient[0]):
                        current_slice = current_ct[slice, int(slice_center)-int(size/2):int(slice_center)+int(size/2), int(slice_center)-int(size/2):int(slice_center)+int(size/2)]
                        current_ct_data[slice, 0, :, :] = current_slice
                else:
                    #If patch_size is bigger than the size of the original pet, make padding with background value as minimum value of the pet
                    for slice in np.array(slices_in_patient[0]):
                        m, n = current_ct[slice, :, :].shape
                        padding = ((max(0, size - m), max(0, size - m)), (max(0, size - n), max(0, size - n)))
                        current_slice = torch.from_numpy(np.pad(current_ct[slice, :, :], padding, 'constant', constant_values=(0, 0)))
                        current_ct_data[slice, 0, :, :] = current_slice

                assert current_data.shape[0] == current_ct_data.shape[0], "The number of slices in the PET and the segmentation are different"
                assert current_data.shape[2] == current_ct_data.shape[2], "The size of the PET and the segmentation are different"
                assert current_data.shape[3] == current_ct_data.shape[3], "The size of the PET and the segmentation are different"

            #----------------- SAVE DATA -----------------

                time_stamp_batch = self.time_stamp.repeat(size*size, 1, 1)
                time_stamp_batch = time_stamp_batch[:, 0, :].permute((1, 0))

                for slice in slices_in_patient[0]:
                    TAC_batch = torch.reshape(current_data[slice, :, :], [current_data.shape[1], size*size])
                    AUC = torch.trapezoid(TAC_batch, time_stamp_batch, dim=0)
                    maskk = AUC > 10
                    maskk = maskk * 1
                    if torch.sum(maskk) <= 0:
                        print("\t\t\tWARNING: Empty slice! Patient: " + str(patient) + " Slice: " + str(slice))
                        continue

                    #----------------- Save data -----------------
                    current_slice = current_data[slice, :, :].type(torch.cuda.FloatTensor)
                    current_seg_slice = current_seg_data[slice, :, :].type(torch.cuda.IntTensor)
                    current_ct_slice = current_ct_data[slice, :, :].type(torch.cuda.FloatTensor)
                    torch.save([patient, slice, current_slice], self.save_data_folder + "/data"+str(patient)+"_"+str(slice)+".pt")
                    torch.save([patient, slice, current_seg_slice],  self.save_data_folder + "/seg" +str(patient)+"_"+str(slice)+".pt")
                    torch.save([patient, slice, current_ct_slice], self.save_data_folder + "/ct" +str(patient)+"_"+str(slice)+".pt")
                    data_list.append(self.save_data_folder+"/data"+str(patient)+"_"+str(slice)+".pt")
                    seg_list.append(self.save_data_folder+"/seg"+str(patient)+"_"+str(slice)+".pt")
                    ct_list.append(self.save_data_folder+"/ct"+str(patient)+"_"+str(slice)+".pt")
        
                print("\t\tImporting Segmentation...")
        
        return [data_list, seg_list, ct_list]
    
    def load_data(self, current_config):
        # Define the location where the dataset is saved
        folder_name = self.dataset_type+"_N"+str(self.current_dataset_size)+"_SEG"+str(len(self.train_config["segmentation_list"]))+"_"
        for seg in current_config["segmentation_list"]:
            folder_name += seg + "_"
        folder_name = folder_name[:-1]

        print("Loading dataset", self.dataset_type, "from", folder_name)

        self.save_data_folder = os.path.join(root_dataset_path, folder_name)

        if not os.path.exists(self.save_data_folder):
            print("WARNING: Dataset files don't exist!")
            return None
        
        print("Patient list: ", self.patient_list, "in", self.dataset_type)

        data_list = list()
        data_seg_list = list()
        data_ct_list = list()
        for patient in self.patient_list:
            patient_files = glob.glob(self.save_data_folder+"/data"+str(patient)+"*.pt")
            patient_seg_files = glob.glob(self.save_data_folder+"/seg"+str(patient)+"*.pt")
            patient_ct_files = glob.glob(self.save_data_folder+"/ct"+str(patient)+"*.pt")

            #Order the files by slice number and patient
            patient_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            patient_seg_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            patient_ct_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

            for file_name in patient_files:
                data_list.append(file_name)
            
            for file_name in patient_ct_files:
                data_ct_list.append(file_name)

            for file_name in patient_seg_files:
                data_seg_list.append(file_name)

        return [data_list, data_seg_list, data_ct_list]
        
        # GOOD TO KNOW: self.save_data_folder and file_name are designed so that different datasets with different patients list, patch size or number of slices are saved separately and not overwritten.
        # This allows to save time when generating bigger datasets!


#----------------- Functions -----------------

def keep_largest_segmentation(input_path, output_path):
    # Load the NIfTI file
    img = nib.load(input_path)
    data = img.get_fdata()

    # Label connected components
    labeled_data, num_features = label(data)

    # Find the largest component
    largest_component = 0
    largest_size = 0
    for feature in range(1, num_features + 1):
        feature_size = np.sum(labeled_data == feature)
        if feature_size > largest_size:
            largest_size = feature_size
            largest_component = feature

    # Keep only the largest component
    if largest_component > 0:
        data[labeled_data != largest_component] = 0

    # Save the resulting segmentation
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, output_path)

def process_files(base_dir):
    # Use glob to find all patient folders matching the pattern
    patient_folders = glob.glob(os.path.join(base_dir, '*DynamicFDG*'))
    
    for patient_folder in patient_folders:
        # Construct the path to the liver.nii.gz file
        input_path = os.path.join(patient_folder, 'NIFTY/Resampled/segmentation/liver.nii.gz')
        
        if os.path.exists(input_path):
            output_path = os.path.join(patient_folder, 'NIFTY/Resampled/segmentation_processed')
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, 'liver.nii.gz')
            keep_largest_segmentation(input_path, output_path)
            print(f"Processed {input_path}")
        else:
            print(f"File not found: {input_path}")


def __main__():
    base_dir = '/home/guests/jorge_padilla/data/DynamicPET_Segmentation'
    process_files(base_dir)

if __name__ == "__main__":
    __main__()