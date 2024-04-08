import os
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.set_root_paths import root_path, root_checkpoints_path
from network import SpaceTempUNet

def train_unet_kinetics(resume_path=None, device=torch.device("cuda:0")):
    """
    Train the UNet model for Dynamic PET reconstruction
    Args:
        resume_path: path to the checkpoint to resume training
    """

    if device == torch.device("cuda:0"):
        num_device = [0]
        accelerator = "gpu"
    else:
        num_device = 1
        accelerator = "cpu"

    # Set up Weights&Biases
    wandb.init(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config_kinetics.yaml"))
    wandb_logger = WandbLogger(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config_kinetics.yaml"))

    # Add device to config
    wandb.config["device"] = device

    # Set up the UNet model
    unet = SpaceTempUNet(wandb.config)

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

    trainer.fit(unet, ckpt_path=resume_path)

    wandb.finish()


def test_unet_kinetics(checkpoint_path, device=torch.device("cuda:0")):
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

    wandb_logger = WandbLogger(project="DynamicPet_segmentation", config=os.path.join(root_path, "config/config_kinetics.yaml"))
    unet = SpaceTempUNet(wandb.config, device=device)

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