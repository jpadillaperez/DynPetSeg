import os
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.set_root_paths import root_path, root_checkpoints_path
from network import SpaceTempUNetSeg1

def train_unet_seg1(kinetic_resume_path=None, resume_path=None, device=torch.device("cuda:0")):
    """ 
    Train the UNet model for Dynamic PET organ segmentation
    Args:
      kinetic_resume_path: path to the checkpoint to resume training of the kinetic model
      resume_path: path to the checkpoint to resume training
    """

    if kinetic_resume_path:
      assert resume_path is None, "Only Kinetic checkpoint or Normal checkpoint should be given, not both."
    if resume_path:
      assert kinetic_resume_path is None, "Only Kinetic checkpoint or Normal checkpoint should be given, not both."


    if device == torch.device("cuda:0"):
      num_device = [0]
      accelerator = "gpu"
    else:
      num_device = 1
      accelerator = "cpu"

    # Set up Weights&Biases
    wandb.init(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config_seg1.yaml"))
    wandb_logger = WandbLogger(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config_seg1.yaml"))

    # Set up the UNet model
    wandb.config["device"] = device

    # Set up the UNet model
    unet = SpaceTempUNetSeg1(wandb.config)

    # Load kinetic weights
    if kinetic_resume_path is not None:
      print("Resuming training from kinetic imaging train: ", kinetic_resume_path)
      weights = torch.load(kinetic_resume_path)
      for name, param in unet.named_parameters():
        if ((name in weights["state_dict"].keys()) and ("final_conv" not in name)):
          param.data = weights["state_dict"][name].data
        else:
          print("Parameter not found: ", name)

    # Load all weights
    if resume_path is not None:
      print("Resuming training from: ", resume_path)
      weights = torch.load(resume_path)
      for name, param in unet.named_parameters():
        if name in weights["state_dict"].keys():
          param.data = weights["state_dict"][name].data
        else:
          print("Parameter not found: ", name)


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
                                        patience=25,
                                        verbose=True, 
                                        mode="min",
                                        check_finite=True
                                        )

    # Trainer
    trainer = pl.Trainer(devices=num_device,
                          accelerator=accelerator,
                          max_epochs=unet.config["epochs"],
                          enable_checkpointing=True,
                          num_sanity_val_steps=1,
                          log_every_n_steps=unet.config["log_freq"],
                          check_val_every_n_epoch=unet.config["val_freq"],
                          callbacks=[checkpoint_callback, early_stop_callback],
                          logger=wandb_logger
                      )

    trainer.fit(unet)
    wandb.finish()


def test_unet_seg1(checkpoint_path, device=torch.device("cuda:0")):
    """
    Test the UNet model for Dynamic PET reconstruction
    Args:
        checkpoint_path: path to the checkpoint to be tested
    """

    print("Testing on", checkpoint_path)

    if device == torch.device("cuda:0"):
      num_device = [0]
      accelerator = "gpu"
    else:
      num_device = 1
      accelerator = "cpu"

    wandb_logger = WandbLogger(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config_seg1.yaml"))
    unet = SpaceTempUNetSeg1(wandb.config, device=device)

    trainer = pl.Trainer(gpus=num_device,
                         accelerator=accelerator,
                         max_epochs=unet.config["epochs"],
                         enable_checkpointing=True,
                         num_sanity_val_steps=1,
                         log_every_n_steps=unet.config["log_freq"],
                         check_val_every_n_epoch=unet.config["val_freq"],
                         logger=wandb_logger,
                        )
    
    trainer.test(unet, ckpt_path=checkpoint_path)
    wandb.finish()