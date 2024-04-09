import wandb
import torch
import yaml
import argparse
from training.training_kinetics import train_unet_kinetics, test_unet_kinetics
from training.training_seg0 import train_unet_seg0, test_unet_seg0
from training.training_seg1 import train_unet_seg1, test_unet_seg1
from training.training_seg2 import train_unet_seg2, test_unet_seg2

#os.environ["WANDB_MODE"] = "offline"
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(mode=True, warn_only=True)
#torch.set_float32_matmul_precision('medium') 

if __name__ == '__main__':

  # Parse command-line arguments
  parser = argparse.ArgumentParser(description='Run experiments')
  parser.add_argument('--experiment', type=str, default='kinetics', choices=['kinetics', 'seg0', 'seg1', 'seg2', 'sweep'],
                      help='Experiment to run (default: kinetics)')
  parser.add_argument('--modality', type=str, default='train', choices=['train', 'test'],
                      help='Training or testing mode (default: train)')
  parser.add_argument('--ckpt', type=str, default=False,
                      help='Checkpoint path for training (default: False)')
  parser.add_argument('--kinetic_ckpt', type=str, default=False,
                      help='Checkpoint path for Kinetics training (default: False)')
  args = parser.parse_args()

  # Initializing device 
  if not torch.cuda.is_available():   
    device = torch.device("cpu")
    print("Using CPU")
  else:
    device = torch.device("cuda:0")
    print("Total GPU Memory {} Gb".format(torch.cuda.get_device_properties(0).total_memory/1e9))

  #Redirecting to appropiate script
  print("Performing", args.modality, "on experiment", args.experiment)

  # Sweep
  if args.modality == "sweep":
    with open("config/sweep_config.yaml", "r") as stream:
      sweep_config = yaml.safe_load(stream)
      sweep_id = wandb.sweep(sweep=sweep_config, project="DynamicPet_segmentation")
    if args.experiment == "kinetics":
      wandb.agent(sweep_id, function=train_unet_kinetics)
    elif args.experiment == "seg0":
      wandb.agent(sweep_id, function=train_unet_seg0)

  # Training or Testing
  else:

    # Kinetics
    if args.experiment == "kinetics":
      if args.modality == "test": 
        test_unet_kinetics(checkpoint_path=args.ckpt, device=device)
      else:
        if args.ckpt:
          train_unet_kinetics(resume_path=args.ckpt, device=device)
        else:
          train_unet_kinetics(device=device)

    # Segmentation 0
    elif args.experiment == "seg0":
      if args.modality == "test": 
        test_unet_seg0(checkpoint_path=args.ckpt, device=device)
      else:
        if args.ckpt and args.kinetic_ckpt:
          train_unet_seg0(kinetic_resume_path=args.kinetic_ckpt, resume_path=args.ckpt, device=device)
        elif args.ckpt:
          train_unet_seg0(resume_path=args.ckpt, device=device)
        elif args.kinetic_ckpt:
          train_unet_seg0(kinetic_resume_path=args.kinetic_ckpt, device=device)
        else:
          train_unet_seg0(device=device)

    # Segmentation 1
    elif args.experiment == "seg1":
      if args.modality == "test": 
        test_unet_seg1(checkpoint_path=args.ckpt, device=device)
      else:
        if args.ckpt and args.kinetic_ckpt:
          train_unet_seg1(kinetic_resume_path=args.kinetic_ckpt, resume_path=args.ckpt, device=device)
        elif args.ckpt:
          train_unet_seg1(resume_path=args.ckpt, device=device)
        elif args.kinetic_ckpt:
          train_unet_seg1(kinetic_resume_path=args.kinetic_ckpt, device=device)
        else:
          train_unet_seg1(device=device)

    # Segmentation 2
    elif args.experiment == "seg2":
      if args.modality == "test": 
        test_unet_seg2(checkpoint_path=args.ckpt, device=device)
      else:
        if args.ckpt and args.kinetic_ckpt:
          train_unet_seg2(kinetic_resume_path=args.kinetic_ckpt, resume_path=args.ckpt, device=device)
        elif args.ckpt:
          train_unet_seg2(resume_path=args.ckpt, device=device)
        elif args.kinetic_ckpt:
          train_unet_seg2(kinetic_resume_path=args.kinetic_ckpt, device=device)
        else:
          train_unet_seg2(device=device)


    else:
      print("Experiment not found")
      exit(1)