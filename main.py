import os
import wandb
import torch
import yaml
import argparse
from ruamel.yaml import YAML
from collections import ChainMap
from training import BaseTraining
from network import Bas1TrainingModule, Bas2TrainingModule, Seg1TrainingModule, Seg2TrainingModule, Seg3TrainingModule, KineticsTrainingModule

#os.environ["WANDB_MODE"] = "offline"
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(mode=True, warn_only=True)
#torch.set_float32_matmul_precision('medium') 

def flatten_yaml_dict(yaml_dict):
    """Flatten the given YAML dictionary by removing the 'value' key."""
    flattened_dict = {}
    for key, val in yaml_dict.items():
        if isinstance(val, dict) and 'value' in val:
          if val['value'] == "None":
            flattened_dict[key] = None
          else:
            flattened_dict[key] = val['value']
        else:
          flattened_dict[key] = val
    return flattened_dict

def merge_dicts(dict1, dict2):
  """Recursively merge two dictionaries."""
  for key, value in dict2.items():
      dict1[key] = value
  return dict1

def load_yaml_file(file_path):
  yaml = YAML()
  with open(file_path, 'r') as file:
    return flatten_yaml_dict(yaml.load(file))


if __name__ == '__main__':
  # Parse command-line arguments
  parser = argparse.ArgumentParser(description='Run experiments')
  parser.add_argument('--experiment', type=str, default='kinetics', choices=['kinetics', 'seg0', 'seg1', 'seg2', 'seg3', 'baseline1', 'baseline2'], required=True,
                      help='Experiment to run (default: kinetics)')
  parser.add_argument('--modality', type=str, default='train', choices=['train', 'test', 'sweep'],
                      help='Training or testing mode (default: train)')
  parser.add_argument('--ckpt', type=str, default=None,
                      help='Checkpoint path for training (default: False)')
  parser.add_argument('--kinetic_ckpt', type=str, default=None,
                      help='Checkpoint path for Kinetics training (default: False)')
  args = parser.parse_args()

  # Load base config  
  base_config = load_yaml_file('/home/guests/jorge_padilla/code/DynamicPET_Segmentation/config/base_config.yaml')
  print("Base config loaded:", base_config)

  #Redirecting to appropiate script
  print("Performing", args.modality, "on experiment", args.experiment)

  # Instantiate BaseTraining
  base = BaseTraining(kinetic_resume_path=args.kinetic_ckpt, full_resume_path=args.ckpt)

  if args.modality == "train":
    config_base_path = "/home/guests/jorge_padilla/code/DynamicPET_Segmentation/config/"
  elif args.modality == "test":
    config_base_path = os.path.dirname(args.ckpt) + "/"
  else:
    print("Modality not found")
    exit(1)

  #---------------- Load config and model ----------------
  if args.experiment == "kinetics":
    config = merge_dicts(base_config, load_yaml_file(config_base_path + "config_kinetics.yaml")) if args.modality == "train" else load_yaml_file(config_base_path + "config.yaml")
    base.load_config(config)
    base.load_model(KineticsTrainingModule)
  elif args.experiment == "seg0":
    config = merge_dicts(base_config, load_yaml_file(config_base_path + "config_seg0.yaml")) if args.modality == "train" else load_yaml_file(config_base_path + "config.yaml")
    base.load_config(config)
    base.load_model(Seg0TrainingModule)
  elif args.experiment == "seg1":
    config = merge_dicts(base_config, load_yaml_file(config_base_path + "config_seg1.yaml")) if args.modality == "train" else load_yaml_file(config_base_path + "config.yaml")
    base.load_config(config)
    base.load_model(Seg1TrainingModule)
  elif args.experiment == "seg2":
    config = merge_dicts(base_config, load_yaml_file(config_base_path + "config_seg2.yaml")) if args.modality == "train" else load_yaml_file(config_base_path + "config.yaml")
    base.load_config(config)
    base.load_model(Seg2TrainingModule)
  elif args.experiment == "seg3":
    config = merge_dicts(base_config, load_yaml_file(config_base_path + "config_seg3.yaml")) if args.modality == "train" else load_yaml_file(config_base_path + "config.yaml")
    base.load_config(config)
    base.load_model(Seg3TrainingModule)
  elif args.experiment == "baseline1":
    config = merge_dicts(base_config, load_yaml_file(config_base_path + "config_baseline1.yaml")) if args.modality == "train" else load_yaml_file(config_base_path + "config.yaml")
    base.load_config(config)
    base.load_model(Bas1TrainingModule)
  elif args.experiment == "baseline2":
    config = merge_dicts(base_config, load_yaml_file(config_base_path + "config_baseline2.yaml")) if args.modality == "train" else load_yaml_file(config_base_path + "config.yaml")
    base.load_config(config)
    base.load_model(Bas2TrainingModule)
  else:
    print("Experiment not found")
    exit(1)

  #---------------- Train or Test ----------------
  if args.modality == "train":
    base.train()

  if args.modality == "test":
    base.test()