import os
import wandb
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.set_root_paths import root_path, root_checkpoints_path
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed as dist

def get_gpu_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


class BaseTraining():
  def __init__(self, kinetic_resume_path=None, full_resume_path=None):
    self.kinetic_resume_path = kinetic_resume_path
    self.full_resume_path = full_resume_path
    self.config = None
    

    if self.kinetic_resume_path:
      assert self.full_resume_path is None, "Only Kinetic checkpoint or Normal checkpoint should be given, not both."
    if self.full_resume_path:
      assert self.kinetic_resume_path is None, "Only Kinetic checkpoint or Normal checkpoint should be given, not both."

    self.model = None

    self.num_devices = torch.cuda.device_count()
    self.accelerator = "gpu"

    self.rank = get_gpu_rank()

  def save_config_as_yaml(self):
      if self.rank == 0:  # Only save on the main process
          output_path = self.config.get("output_checkpoint_path")
          if output_path:
              os.makedirs(output_path, exist_ok=True)
              config_path = os.path.join(output_path, "config.yaml")
              with open(config_path, 'w') as yaml_file:
                  yaml.dump(self.config, yaml_file, default_flow_style=False)
              print(f"Config saved to {config_path}")
          else:
              print("Warning: output_checkpoint_path not set in config. Config not saved.")

  def load_config(self, config: dict):
    self.config = config
    self.config["kinetic_resume_path"] = self.kinetic_resume_path
    self.config["full_resume_path"] = self.full_resume_path
    self.config["rank"] = self.rank
    self.config["num_devices"] = self.num_devices
    self.config["static_graph"] = False if self.config.get("alternate_training", False) else True

    # Set up Weights&Biases
    if self.rank == 0:
      wandb.init(project="DynamicPet_segmentation", config=self.config)
      self.wandb_logger = WandbLogger(project="DynamicPet_segmentation", config=self.config)      
      self.config["run_name"] = wandb.run.name
      self.config["output_checkpoint_path"] = os.path.join(root_checkpoints_path, "checkpoints", wandb.run.name)

      # Call the function to save the config
      self.save_config_as_yaml()



  def load_model(self, model_type):
    self.model = model_type(self.config)
    

  def train(self):
      """ 
      Train the model
      """
      # Callbacks
      checkpoint_path = os.path.join(root_checkpoints_path, "checkpoints", wandb.run.name)
      print("Checkpoints will be saved in: ", checkpoint_path)

      checkpoint_callback = ModelCheckpoint(monitor=self.config["monitor_metric"],
                                            dirpath=checkpoint_path,
                                            save_top_k=1,
                                            mode="min",
                                            every_n_epochs=5,
                                            save_last=True
                                            )

      early_stop_callback = EarlyStopping(monitor=self.config["monitor_metric"],
                                          min_delta=0, 
                                          patience=25,
                                          verbose=True, 
                                          mode="min",
                                          check_finite=True
                                          )
      # Trainer
      if self.num_devices == 1:
        print("Using Single GPU strategy")
        trainer = pl.Trainer(devices=self.num_devices,
                              accelerator=self.accelerator,
                              max_epochs=self.model.config["epochs"],
                              enable_checkpointing=True,
                              num_sanity_val_steps=1,
                              check_val_every_n_epoch=self.model.config["val_freq"],
                              callbacks=[checkpoint_callback, early_stop_callback],
                              logger=self.wandb_logger
                          )
      else:
        print("Using DDP strategy")
        trainer = pl.Trainer(devices=self.num_devices,
                        accelerator=self.accelerator,
                        max_epochs=self.model.config["epochs"],
                        #precision=16,
                        #accumulate_grad_batches=4,
                        enable_checkpointing=True,
                        strategy=DDPStrategy(static_graph=self.config["static_graph"], find_unused_parameters=not self.config["static_graph"]),
                        num_sanity_val_steps=1,
                        check_val_every_n_epoch=self.model.config["val_freq"],
                        callbacks=[checkpoint_callback, early_stop_callback],
                        logger=self.wandb_logger
                    )


      trainer.fit(self.model)
      wandb.finish()


  def test(self):
      """
      Test the UNet model for Dynamic PET reconstruction
      """

      print("Testing on", self.full_resume_path)

      trainer = pl.Trainer(devices=self.num_devices,
                          accelerator=self.accelerator,
                          max_epochs=self.model.config["epochs"],
                          enable_checkpointing=True,
                          num_sanity_val_steps=1,
                          check_val_every_n_epoch=self.model.config["val_freq"],
                          logger=self.wandb_logger
                          )
      
      trainer.test(self.model, ckpt_path=self.full_resume_path)
      wandb.finish()