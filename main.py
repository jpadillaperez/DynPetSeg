import os
import wandb
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.set_root_paths import root_path, root_checkpoints_path
from network import SpaceTempUNet, SpaceTempUNetSeg

#os.environ["WANDB_MODE"] = "offline"
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(mode=True, warn_only=True)
#torch.set_float32_matmul_precision('medium') #APUNTAR EN NOTES QUE ESTO ES HORROROSO

if not torch.cuda.is_available():   
  current_device = 1
  accelerator = "cpu"
  print("*** ERROR: no GPU available ***")
else:
  current_device = [0]
  accelerator = "gpu"
  print("Total GPU Memory {} Gb".format(torch.cuda.get_device_properties(0).total_memory/1e9))
#----------------------------------------------

def train_unet(resume_path=None, enable_testing=False):
  """
  Train the UNet model for Dynamic PET reconstruction
  Args:
    resume_path: path to the checkpoint to resume training
    enable_testing: if True, test the model at the end of the training
  """

  # Set up Weights&Biases
  wandb.init(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config.yaml"))
  wandb_logger = WandbLogger(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config.yaml"))

  # Set up the UNet model
  unet = SpaceTempUNet(wandb.config)

  checkpoint_path = os.path.join(root_checkpoints_path, "checkpoints", wandb.run.name)
  print("Checkpoints will be saved in: ", checkpoint_path)
  checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                        dirpath=checkpoint_path,
                                        save_top_k=1,
                                        mode="min",
                                        every_n_epochs=5,
                                        save_last=True
                                        )

  early_stop_callback = EarlyStopping(monitor="val_loss", 
                                      min_delta=0, 
                                      patience=25,
                                      verbose=True, 
                                      mode="min",
                                      check_finite=True
                                      )
  
  trainer = pl.Trainer(devices=current_gpu,
                        accelerator=accelerator,
                        max_epochs=unet.config["epochs"],
                        enable_checkpointing=True,
                        num_sanity_val_steps=1,
                        log_every_n_steps=unet.config["log_freq"],
                        check_val_every_n_epoch=unet.config["val_freq"],
                        callbacks=[checkpoint_callback, early_stop_callback],
                        logger=wandb_logger
                    )

  trainer.fit(unet, ckpt_path=resume_path)

  if enable_testing:
    trainer.test(ckpt_path="best")
    trainer.test(ckpt_path="last")

  wandb.finish()

#----------------------------------------------

def train_seg_unet(kinetic_resume_path=None, resume_path=None, enable_testing=False):
  """ Train the UNet model for Dynamic PET organ segmentation
  Args:
    kinetic_resume_path: path to the checkpoint to resume training of the kinetic model
    resume_path: path to the checkpoint to resume training
    enable_testing: if True, test the model at the end of the training
  """

  # Set up Weights&Biases
  wandb.init(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config.yaml"))
  wandb_logger = WandbLogger(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config.yaml"))

  # Set up the UNet model
  unet = SpaceTempUNetSeg(wandb.config)

  # Load all weights
  if resume_path is not None:
    print("Resuming training from: ", resume_path)
    weights = torch.load(resume_path)
    for name, param in unet.named_parameters():
      if name in weights["state_dict"].keys():
        param.data = weights["state_dict"][name].data
      else:
        print("Parameter not found: ", name)

  # Load kinetic weights
  if kinetic_resume_path is not None:
    print("Resuming training from kinetic imaging train: ", kinetic_resume_path)
    weights = torch.load(kinetic_resume_path)
    #Freeze all weights except the last layers
    for name, param in unet.named_parameters():
      if ((name in weights["state_dict"].keys()) and ("final_conv" not in name)):
        param.data = weights["state_dict"][name].data
        param.requires_grad = False
      else:
        print("Parameter not found: ", name)
      #Unfreeze last layers
      if "final_conv_1" in name or "final_conv_2" in name:
        param.requires_grad = True
        print("Unfreezing: ", name)
      

  # Callbacks
  checkpoint_path = os.path.join(root_checkpoints_path, "checkpoints", wandb.run.name)
  print("Checkpoints will be saved in: ", checkpoint_path)

  checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                        dirpath=checkpoint_path,
                                        save_top_k=1,
                                        mode="min",
                                        every_n_epochs=5,
                                        save_last=True
                                        )

  early_stop_callback = EarlyStopping(monitor="val_loss", 
                                      min_delta=0, 
                                      patience=50,
                                      verbose=True, 
                                      mode="min",
                                      check_finite=True
                                      )
  
  # Trainer
  trainer = pl.Trainer(devices=current_device,
                        accelerator=accelerator,
                        max_epochs=unet.config["epochs"],
                        enable_checkpointing=True,
                        num_sanity_val_steps=1,
                        log_every_n_steps=unet.config["log_freq"],
                        check_val_every_n_epoch=unet.config["val_freq"],
                        callbacks=[checkpoint_callback, early_stop_callback],
                        logger=wandb_logger
                    )

  trainer.fit(unet, ckpt_path=None)

  if enable_testing:
    trainer.test(ckpt_path="best")
    trainer.test(ckpt_path="last")

  wandb.finish()

#----------------------------------------------


def test_unet(checkpoint_path):
  """
  Test the UNet model for Dynamic PET reconstruction
  Args:
    checkpoint_path: path to the checkpoint to be tested
  """

  print("Testing on", checkpoint_path)

  wandb_logger = WandbLogger(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config.yaml"))
  unet = SpaceTempUNet(wandb.config)

  trainer = pl.Trainer(gpus=current_gpu,
                        max_epochs=unet.config["epochs"],
                        enable_checkpointing=True,
                        num_sanity_val_steps=1,
                        log_every_n_steps=unet.config["log_freq"],
                        check_val_every_n_epoch=unet.config["val_freq"],
                        logger=wandb_logger,
                      )
  
  trainer.test(unet, ckpt_path=checkpoint_path)
  wandb.finish()

#----------------------------------------------

if __name__ == '__main__':

  with open("config/config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

  print("Performing ", config["modality"]["value"])

  # Test
  if config["modality"]["value"] == "test": 
    test_unet(config["saved_checkpoint"]["value"])

  # Train
  elif config["modality"]["value"] == "train" or config["modality"]["value"] == "overfit":
    if config["continue_checkpoint"]["value"] == "":
      train_unet()
    else:
      train_unet(resume_path=config["continue_checkpoint"]["value"], enable_testing=False)

  # Train Segmentation
  elif config["modality"]["value"] == "train_seg" or config["modality"]["value"] == "overfit_seg":
    if config["continue_checkpoint"]["value"] == "":
      train_seg_unet()
    elif config["kinetic_checkpoint"]["value"] != "":
      train_seg_unet(kinetic_resume_path=config["kinetic_checkpoint"]["value"], enable_testing=False)
    else:
      train_seg_unet(resume_path=config["continue_checkpoint"]["value"], enable_testing=True)

  # Sweep
  elif config["modality"]["value"] == "sweep":
    with open("config/sweep_config.yaml", "r") as stream:
      sweep_config = yaml.safe_load(stream)
      sweep_id = wandb.sweep(sweep=sweep_config, project="DynamicPet_segmentation")
    wandb.agent(sweep_id, function=train_unet)

  else:
    print("ERROR: modality not recognized!")

