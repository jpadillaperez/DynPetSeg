import os

root_path = "/home/guests/jorge_padilla/code/DynamicPET_Segmentation/"
print("root_path =",root_path)

#root_checkpoints_path = "/home/guests/jorge_padilla/output/models/DynamicPET"
root_checkpoints_path = "/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation"
print("root_checkpoints_path =",root_checkpoints_path)

#root_data_path = "/home/polyaxon-data/data1/DynamicPET"
root_data_path = "/home/polyaxon-data/data1/DynamicPET_segmentation"
print("root_data_path =", root_data_path)

#root_dataset_path = "/home/polyaxon-data/data1/DynamicPET/dataset_release"
#root_dataset_path = "/home/polyaxon-data/data1/DynamicPET/dataset"
root_dataset_path = "/home/guests/jorge_padilla/data/DynamicPET_Segmentation"
print("root_dataset_path =", root_dataset_path)

checkpoint_path = os.path.join(root_checkpoints_path, "checkpoints", "zesty-eon-22", "last.ckpt")
if checkpoint_path is not None:
    print("Using previous checkpoint =", checkpoint_path)