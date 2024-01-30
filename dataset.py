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
from pympler import asizeof

class DynPETDataset(CacheDataset):
    def __init__(self, config, dataset_type): 
        # Enforce determinism
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # Read global config
        self.config = config
        self.dataset_type = dataset_type    #train, validation or test
        self.patch_size = self.config["patch_size"]

        # Create config for each dataset type from global config
        if (self.config["slices_per_patient_train"] == "None") or (self.config["slices_per_patient_train"] == "all"):
            self.train_config = {"patient_list": self.config["patient_list"]["train"], "slices_per_patient": "all", "bounding_box": self.config["bounding_box"]}
        else:
            self.train_config = {"patient_list": self.config["patient_list"]["train"], "slices_per_patient": self.config["slices_per_patient_train"], "bounding_box": self.config["bounding_box"]}

        if (self.config["slices_per_patient_val"] == "None") or (self.config["slices_per_patient_val"] == "all"):
            self.val_config = {"patient_list": self.config["patient_list"]["validation"], "slices_per_patient": "all", "bounding_box": self.config["bounding_box"]}
        else:
            self.val_config = {"patient_list": self.config["patient_list"]["validation"], "slices_per_patient": self.config["slices_per_patient_val"], "bounding_box": self.config["bounding_box"]}

        if (self.config["slices_per_patient_test"] == "None") or (self.config["slices_per_patient_test"] == "all"):
            self.test_config = {"patient_list": self.config["patient_list"]["test"], "slices_per_patient": "all", "bounding_box": self.config["bounding_box"]}
        else: 
            self.test_config = {"patient_list": self.config["patient_list"]["test"], "slices_per_patient": self.config["slices_per_patient_test"], "bounding_box": self.config["bounding_box"]}

        # Select the correct config
        self.idif = dict()
        self.data = list()
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
        #TODO: load torch from here #DONE
        item = torch.load(self.data[idx])
        item[2] = item[2].to(torch.float16).type(torch.cuda.FloatTensor)
        
        #check if item[2] is empty
        if torch.sum(item[2]) == 0:
            print("WARNING: Empty slice!")
            
        return item

    def __len__(self):
        return int(self.length)
    
    def build_dataset(self, current_config):
        self.patient_list = current_config["patient_list"]
        self.slices_per_patient = dict()
        for p in self.patient_list:
            self.slices_per_patient[p] = dict()
            self.load_txt_data(p)

        #FIXME: calculate current_dataset_size properly with for loop over patients #FIXED
        #self.current_dataset_size = current_config["slices_per_patient"] * len(current_config["patient_list"])     
        current_size = 0
        for patient in self.patient_list:
            for pet in self.slices_per_patient[patient]:
                current_size += int(self.slices_per_patient[patient][pet]["Slice_Count"])
        self.current_dataset_size = current_size

        # Load existing data, if possible
        load_data = self.load_data() #FIXME: update load data
        if load_data is None:  
            # If the dataset does not exist, create the folder where it will be saved
            if not os.path.exists(self.save_data_folder):
                os.makedirs(self.save_data_folder)

            self.data = self.read_dynpet()
            print("Dataset", self.dataset_type, "was saved in", self.save_data_folder)
        else:                   
            self.data = load_data
        
        self.length = len(self.data)

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

        #SaVe number of slices in dict
        data = pd.read_csv(info_txt_path, sep=",")
        for i in range(len(data)):
            self.slices_per_patient[patient][data["PET"][i]] = {"Slice_Count": data["Slice_Count"][i], "Slice_Size": data["Slice_Size"][i]}

        return 

    def read_dynpet(self): 
        print("Creating dataset", self.dataset_type, "with", self.current_dataset_size, "slices...")
        data_list = list()
        for patient in self.patient_list:
            print("\tProcessing patient: " + str(patient))
            patient_folder = glob.glob(os.path.join(root_data_path, "*DynamicFDG_"+patient))[0]

            # When using self.config["slices_per_patient_*"] --> probably the selection of self.slice can be shortened a little bit
            #self.slices_per_patients = int(self.current_dataset_size / len(self.patient_list)) #FIXME: Make a dictionary with patients and slices

            #DOUBT JPP: When using all of the slices per patient, it still means same number of slices in all the patients? --> No

            if self.current_dataset_size == 1: 
                # When using only one slice per patient (for example during debugging), the location of the slice is hard-coded (it should select a slice with kidneys)
                self.slices = [212]
                print("\tN_slices=" + str(len(self.slices)) + "/1 ; slices:", self.slices)
            else: 
                # When config["slices_per_patient_*"]>1 (and therefore self.current_dataset_size > 1), the slices are selected within a ROI defined by a bouding box (bb). We used a bb
                # including the lungs and the bladder. In this way we didn't considered the head (because of the movement over the acquisition) and the legs (which are not very informative).
                # The use of the bb is not mandatory.  

                #TODO: Add a config flag to avoid this bbox step #DONE
                if self.train_config["bounding_box"]:
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

                    # Final selection of the pick 
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
                                #cut patch size from center of slice
                                current_slice = current_pet[slice, int(slice_center)-int(size/2):int(slice_center)+int(size/2), int(slice_center)-int(size/2):int(slice_center)+int(size/2)] 
                        else:
                            for j in range(len(self.slices)): 
                                slice = int(self.slices[j])
                                #TODO: add padding with background value as minimum value of the pet
                                m, n = current_pet[slice, :, :].shape
                                padding = ((max(0, size - m), max(0, size - m)), (max(0, size - n), max(0, size - n)))
                                current_slice = np.pad(current_pet[slice, :, :], padding, 'constant', constant_values=(0, 0))
                                current_data[j, i, :, :] = current_slice/1000           # from Bq/ml to kBq/ml

                else:
                    print("\t\tIgnoring bounding box..." )

                    #TODO: Use the number of slices per patient and per pet #DONE
                    slices_in_patient = list()
                    sizes_in_patient = list()
                    pet_list = self.slices_per_patient[patient].keys()
                    for pet in self.slices_per_patient[patient]:
                        sizes_in_patient.append(int(self.slices_per_patient[patient][pet]["Slice_Size"]))
                        slices_in_patient.append(int(self.slices_per_patient[patient][pet]["Slice_Count"]))

                    print("\t\tPets in patient: " + str(pet_list))
                    print("\t\tSlices in patient: " + str(slices_in_patient))
                    print("\t\tSizes in patient: " + str(sizes_in_patient))
                    
                    if self.patch_size is None:
                        size = max(sizes_in_patient)
                    else:
                        size = self.patch_size

                    slices = max(slices_in_patient)

                    #current_data contains the dynpet for just one patient
                    current_data = torch.zeros((slices, len(pet_list), size, size))  #[SLICES, DYNPET, X, Y] TODO: better [DYNPET, SLICES, X, Y]?         

                    #Iterate over all the pets of the dynamic pet of the same patients with progress bar
                    for i, p in enumerate(tqdm(pet_list, desc="Processing pets")):
                        current_pet = torch.from_numpy(np.int32(sitk.GetArrayFromImage(sitk.ReadImage(patient_folder + "/NIFTY/Resampled/PET_" + str(p) + ".nii.gz"))))

                        if size < current_pet[0, :, :].shape[0] or size < current_pet[0, :, :].shape[1]:
                            #DONE: If patch_size is smaller than the size of the original pet, make center crop
                            slice_center = torch.tensor(current_pet[0, :, :].shape)[0] / 2
                            for slice in range(slices_in_patient[i]): 
                                current_slice = current_pet[slice, int(slice_center)-int(size/2):int(slice_center)+int(size/2), int(slice_center)-int(size/2):int(slice_center)+int(size/2)]
                                current_data[slice, i, :, :] = current_slice/1000           # from Bq/ml to kBq/ml
                        else:           
                            #DONE: If patch_size is bigger than the size of the original pet, make padding with background value as minimum value of the pet
                            for slice in range(slices_in_patient[i]):
                                m, n = current_pet[slice, :, :].shape
                                padding = ((max(0, size - m), max(0, size - m)), (max(0, size - n), max(0, size - n)))
                                current_slice = torch.from_numpy(np.pad(current_pet[slice, :, :], padding, 'constant', constant_values=(0, 0)))
                                current_data[slice, i, :, :] = current_slice/1000           # from Bq/ml to kBq/ml

                    time_stamp_batch = self.time_stamp.repeat(size*size, 1, 1)
                    time_stamp_batch = time_stamp_batch[:, 0, :].permute((1, 0))

                    for j in range(slices):
                        TAC_batch = torch.reshape(current_data[j, :, :], [current_data.shape[1], size*size])
                        AUC = torch.trapezoid(TAC_batch, time_stamp_batch, dim=0)

                        maskk = AUC > 10
                        maskk = maskk * 1

                        #----------------- Save data -----------------

                        if torch.sum(maskk) > 0:
                            current_slice = current_data[j, :, :].to(torch.float16).type(torch.cuda.FloatTensor)
                            torch.save([patient, j, current_slice], self.save_data_folder+"/data"+str(patient)+"_"+str(j)+".pt")
                            data_list.append(self.save_data_folder+"/data"+str(patient)+"_"+str(j)+".pt")
                        else:
                            print("WARNING: Empty slice! Patient: " + str(patient) + " Slice: " + str(j))


            ###TODO: Needed for segmentation
            ### Load label map
            ### label_path = patient_folder+"/NIFTY/Resampled/labels.nii.gz"
            ### current_label_data = torch.zeros((len(self.slices), size, size))          
            ### label_ = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))
            ### slice_size = label_[0, :, :].shape
            ### slice_center = torch.tensor(slice_size)[0] / 2
            ### for j in range(len(self.slices)): 
            ###     slice = int(self.slices[j])
            ###     current_slice = label_[slice, int(slice_center)-int(size/2):int(slice_center)+int(size/2), int(slice_center)-int(size/2):int(slice_center)+int(size/2)]                
            ###     current_label_data[j, :, :] = current_slice
            ### for j in range(len(self.slices)):
            ###     data.append([patient, self.slices[j], current_data[j, :, :], current_label_data[j, :, :]])

            
            #torch.save(data, self.save_data_folder+"/data"+str(patient)+".pt")
            #data_list.append(self.save_data_folder+"/data"+str(patient)+".pt")


            #TODO: Clear variables storing the patients #DONE
            #del current_data
            #del data

        #FIXME: it was training without slices number #DONE
        #FIXME: self.data is using a lot of memory #DONE
        
        return data_list
    
    def load_data(self):
        # Define the location where the dataset is saved
        folder_name = self.dataset_type+"_N"+str(self.current_dataset_size)+"_P"+str(self.patch_size)
        self.save_data_folder = os.path.join(root_dataset_path, folder_name)

        # If the dataset exists, load it and return it

        #FIXME: JUST SAVE THE LOCATION OF EACH FILE #DONE

        data_list = list()
        for patient in self.patient_list:
            patient_files = glob.glob(self.save_data_folder+"/data"+str(patient)+"*.pt")
            if len(patient_files) == 0:
                print("WARNING: Dataset files don't exist!")
                return None
            else:
                for file_name in patient_files:
                    data_list.append(file_name)

        return data_list

        #file_name = "data"+str(self.patient_list)+".pt"
        #if os.path.exists(self.save_data_folder+"/" + file_name):
        #    data = torch.load(self.save_data_folder+"/"+file_name) #FIXME: Just save the location of files, not the complete set of examples
        #    return data
        #else:
        #    print("\tWARNING: " + file_name + " doesn't exist!")
        #    return None
        
        # GOOD TO KNOW: self.save_data_folder and file_name are designed so that different datasets with different patients list, patch size or number of slices are saved separately and not overwritten.
        # This allows to save time when generating bigger datasets!