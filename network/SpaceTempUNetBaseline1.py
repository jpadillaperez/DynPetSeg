import os
import torch
import wandb
import numpy as np
import pandas as pd
import multiprocessing
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import pearsonr
from monai.losses import DiceLoss

from dataset import DynPETDataset
from network.unet_blocks_2D import UNet_2D
from utils.utils_kinetic import PET_2TC_KM_batch
from utils.utils_logging import log_slice, log_curves, mask_data
import utils.similaritymeasures_torch as similaritymeasures_torch
from utils.utils_main import make_save_folder_struct, reconstruct_prediction, apply_final_activation
from utils.utils_torch import torch_interp_Nd, WarmupScheduler, weights_init_kaiming, weights_init_xavier
#----------------------------------------------

class SpaceTempUNetBaseline1(pl.LightningModule):
  def __init__(self, config):
    # Enforce determinism
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    
    super(SpaceTempUNetBaseline1, self).__init__()

    # Read configuration file and add new info if needed
    self.config = config
    self.config["output_size"] = len(self.config["segmentation_list"]) + 1  # This is the number of output channels in kinetic network (1 per kinetic parameter)
    self.config["mask_loss"] = True

    print("\nConfiguration: ", self.config)

    # Network 2 Decoder Heads
    self.model = UNet_2D(in_channels=64, out_channels=self.config["output_size"], config=self.config)
    
    # Initialize weights
    if self.config["weight_init"] == "kaiming":
      self.model.apply(weights_init_kaiming)
    elif self.config["weight_init"] == "xavier":
      self.model.apply(weights_init_xavier)

    # Loss function
    self.dice_loss = DiceLoss(reduction="mean", softmax=True, include_background=True)
    self.softmax = torch.nn.Softmax(dim=1) #Just for the test_step

    frame_duration = [10, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 30, 30, 60, 60, 60, 60, 120, 120, 120, 120, 120, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    self.frame_duration = np.array(frame_duration) / 60  # from s to min

    self.validation_step_outputs = []

    self.num_val_log_images = 0
    self.num_train_log_images = 0

  def setup(self, stage): 
    self.stage = stage
    
    if stage == "fit":
      self.train_dataset = DynPETDataset(self.config, "train", patch_size=self.config["patch_size"])
      if self.config["remove_slices_without_segmentation"]:
        self.train_dataset.remove_slices_without_segmentation()
      self.idif_train_set = self.train_dataset.idif

      self.val_dataset = DynPETDataset(self.config, "validation", patch_size=self.train_dataset.__get_patch_size__())
      if self.config["remove_slices_without_segmentation"]:
        self.val_dataset.remove_slices_without_segmentation()
      self.idif_val_set = self.val_dataset.idif

      if self.config["overfit"]:
        self.val_dataset = self.train_dataset

      self.patch_size = self.train_dataset.__get_patch_size__()

      self.t = self.train_dataset.t.to(self.config["device"])
      self.time_stamp = self.train_dataset.time_stamp.to(self.config["device"])
      self.frame_duration_batch = torch.from_numpy(np.array(self.frame_duration)).unsqueeze(-1).repeat(1, self.patch_size*self.patch_size).to(self.config["device"])
      
      self.t_batch = self.t.repeat(self.patch_size*self.patch_size, 1, 1)
      self.time_stamp_batch = self.time_stamp.repeat(self.patch_size*self.patch_size, 1, 1)

    if stage == "test":
      self.test_dataset = DynPETDataset(self.config, "test")
      self.patch_size = self.test_dataset.__get_patch_size__()
      self.idif_test_set = self.test_dataset.idif
      self.t = self.test_dataset.t.to(self.config["device"])
      self.time_stamp = self.test_dataset.time_stamp.to(self.config["device"])
      self.t_batch = self.t.repeat(self.patch_size*self.patch_size, 1, 1)
      self.time_stamp_batch = self.time_stamp.repeat(self.patch_size*self.patch_size, 1, 1)

  def forward(self, x):
    return self.model(x)

  def loss_function(self, pred_seg, real_seg):
    return self.dice_loss(pred_seg, real_seg)


  def train_dataloader(self):
      #num_cpus = multiprocessing.cpu_count()
      num_cpus = 0
      train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=num_cpus)
      return train_loader

  def val_dataloader(self):
      #num_cpus = multiprocessing.cpu_count()
      num_cpus = 0
      val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=num_cpus)
      return val_loader
  
  def test_dataloader(self):
      #num_cpus = multiprocessing.cpu_count()
      num_cpus = 0
      # If batch_size!=1 test_set and test_epoch_end may not work as expected
      test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=num_cpus)
      return test_loader


  def training_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1)  # adding channel dimension --> [b, 1, 62, w, h]
    x = F.pad(TAC_mes_batch, (0,0,0,0,1,1))   # padding --> [b, 1, 64, w, h]
    truth_seg = F.one_hot(batch[3].long(), num_classes= len(self.config["segmentation_list"]) + 1).permute(0, 3, 1, 2).float()  # [b, SEG, w, h]
    x = x[:, 0, :, :, :]  # [b, 62, w, h]

    #------------------ Forward pass ------------------
    seg_logits = self.forward(x)        # [b, seg, 1, w, h]

    #------------------ Process Segmentation Output ------------------
    seg_output = self.softmax(seg_logits)
    seg_output = torch.argmax(seg_output, dim=1)

    #------------------ Loss ------------------
    loss = self.loss_function(seg_logits, truth_seg)

    #------------------ Log the losses ------------------
    self.log('train_loss', loss, on_step=False, on_epoch=True)
    
    #------------------ Log the learning rate ------------------
    self.log('Learning_Rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #------------------ Log the figures ------------------
    if ((np.random.rand() < 0.5) and (self.num_train_log_images < self.config["log_train_imgs"])):
      #------------------ Log Segmentation ------------------
      fig, axes = plt.subplots(2, 2, figsize=(10, 10))
      fig.suptitle("Patient: "+str(batch[0][0])+" Slice: "+str(batch[1][0]))
      #First subplot
      im1 = axes[0, 0].imshow(batch[2][0, 30, :, :].cpu().detach().numpy(), cmap="gray")
      axes[0, 0].axis("off")
      axes[0, 0].set_title("30t")
      im1.set_clim(0, np.max(batch[2][0, 30, :, :].cpu().detach().numpy()))
      #Second subplot
      im2 = axes[1, 0].imshow(batch[2][0, 50, :, :].cpu().detach().numpy(), cmap="gray")
      axes[1, 0].axis("off")
      axes[1, 0].set_title("50t")
      im2.set_clim(0, np.max(batch[2][0, 50, :, :].cpu().detach().numpy()))
      #Rest of the subplots
      axes[0, 1].imshow(seg_output[0, :, :].cpu().detach().numpy(), cmap="binary")
      axes[0, 1].axis("off")
      axes[0, 1].set_title("prediction")
      axes[1, 1].imshow(truth_seg[0, 1, :, :].cpu().detach().numpy(), cmap="binary")
      axes[1, 1].axis("off")
      axes[1, 1].set_title("Truth")
      plt.tight_layout()
      wandb.log({"Segmentation (training batch)": wandb.Image(fig)})
      plt.close()

      self.num_train_log_images += 1

    return {"loss": loss}

  def on_train_epoch_end(self):
    self.num_train_log_images = 0
    return 

  def validation_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
    x = torch.nn.functional.pad(TAC_mes_batch, (0,0,0,0,1,1), "replicate")   # padding --> [b, 1, 64, w, h]
    truth_seg = F.one_hot(batch[3].long(), num_classes= len(self.config["segmentation_list"]) + 1).permute(0, 3, 1, 2).float()  # [b, SEG, w, h]
    x = x[:, 0, :, :, :]  # [b, 62, w, h]

    #------------------ Forward pass ------------------
    seg_logits = self.forward(x)        # [b, seg, 1, w, h]

    #------------------ Process Segmentation Output ------------------
    seg_output = self.softmax(seg_logits)
    seg_output = torch.argmax(seg_output, dim=1)

    #------------------ Loss ------------------
    loss = self.loss_function(seg_logits, truth_seg)

    #----------------- Log the losses ------------------
    self.log('val_loss', loss, on_step=False, on_epoch=True)

    #------------------ Log the figures ------------------
    if ((np.random.rand() < 0.5) and (self.num_val_log_images < self.config["log_val_imgs"])):
      #------------------ Log Segmentation ------------------
      fig, axes = plt.subplots(2, 2, figsize=(10, 10))
      fig.suptitle("Patient: "+str(batch[0][0])+" Slice: "+str(batch[1][0]))
      #First subplot
      im1 = axes[0, 0].imshow(batch[2][0, 30, :, :].cpu().detach().numpy(), cmap="gray")
      axes[0, 0].axis("off")
      axes[0, 0].set_title("30t")
      im1.set_clim(0, np.max(batch[2][0, 30, :, :].cpu().detach().numpy()))
      #Second subplot
      im2 = axes[1, 0].imshow(batch[2][0, 50, :, :].cpu().detach().numpy(), cmap="gray")
      axes[1, 0].axis("off")
      axes[1, 0].set_title("50t")
      im2.set_clim(0, np.max(batch[2][0, 50, :, :].cpu().detach().numpy()))
      #Rest of the subplots
      axes[0, 1].imshow(seg_output[0, :, :].cpu().detach().numpy(), cmap="binary")
      axes[0, 1].axis("off")
      axes[0, 1].set_title("prediction")
      axes[1, 1].imshow(truth_seg[0, 1, :, :].cpu().detach().numpy(), cmap="binary")
      axes[1, 1].axis("off")
      axes[1, 1].set_title("Truth")
      plt.tight_layout()
      fig_seg = wandb.log({"Segmentation (validation batch)": wandb.Image(fig)})
      plt.close()

      self.num_val_log_images += 1

      self.validation_step_outputs.append({"fig_seg": fig_seg})
      return {'val_loss': loss, "fig_seg": fig_seg}
    else:
      self.validation_step_outputs.append({"val_loss": loss, "fig_seg": None})
      return {'val_loss': loss, "fig_seg": None}
  
  def on_validation_epoch_end(self):
    self.validation_step_outputs.clear()
    self.num_val_log_images = 0
    return
  
  def test_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    patients_in_batch = batch[0]
    slices_in_batch = batch[1]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
    x = torch.nn.functional.pad(TAC_mes_batch, (0,0,0,0,1,1), "replicate")   # padding --> [b, 1, 64, w, h]

    #------------------ Forward pass ------------------
    seg_logits = self.forward(x)        # [b, seg, 1, w, h]

    #------------------ Process Segmentation Output ------------------
    seg_logits = seg_logits[:, :, 0, :, :]        # [b, Seg, w, h]
    seg_output = self.softmax(seg_logits)
    seg_output = torch.argmax(seg_output, dim=1)

    #------------------ Save the results locally ------------------
    current_run_name = wandb.run.name
    resume_run_name = os.path.split(os.path.split(self.trainer.ckpt_path)[0])[1]
    self.img_path, self.pd_path, self.pt_path, self.nifty_path = make_save_folder_struct(current_run_name, resume_run_name, root_checkpoints_path, self.trainer.ckpt_path)
    to_save = [patients_in_batch, slices_in_batch, seg_output]
    torch.save(to_save, os.path.join(self.pt_path, "P_"+str(patients_in_batch[0])+"_B_"+str(batch_idx)+".pt"))

    #------------------ Log the figures ------------------
    if batch_idx % 50 == 0:
      # Log TAC             
      log_curves(TAC_mes_batch[:, :, 0:62, :, :].cpu().detach().numpy(), TAC_pred_batch.cpu().detach().numpy(), self.time_stamp.to("cpu"), self.time_stamp.to("cpu"), self.current_epoch)
      plt.savefig(os.path.join(self.img_path, "TAC_P_"+str(patients_in_batch[0])+"_B_"+str(batch_idx)+".png"))
      plt.close()

      # Log segmentation
      fig, axes = plt.subplots(2, len(self.config["segmentation_list"]) + 2, figsize=((len(self.config["segmentation_list"]) + 2) * 2, len(self.config["segmentation_list"]) + 2))  
      fig.suptitle("Patient: "+str(patients_in_batch[0])+" Slice: "+str(slices_in_batch[0]))
      #First subplot
      im1 = axes[0, 0].imshow(batch[2][0, 30, :, :].cpu().detach().numpy(), cmap="gray")
      axes[0, 0].axis("off")
      axes[0, 0].set_title("30t")
      im1.set_clim(0, np.max(batch[2][0, 30, :, :].cpu().detach().numpy()))
      #Second subplot
      im2 = axes[1, 0].imshow(batch[2][0, 50, :, :].cpu().detach().numpy(), cmap="gray")
      axes[1, 0].axis("off")
      axes[1, 0].set_title("50t") 
      im2.set_clim(0, np.max(batch[2][0, 50, :, :].cpu().detach().numpy()))
      #Rest of the subplots
      for i in range(0, (len(self.config["segmentation_list"]) + 1), 1):
        axes[0, i+1].imshow(seg_output[0, :, :].cpu().detach().numpy(), cmap="binary")
        axes[0, i+1].axis("off")
        title = self.config["segmentation_list"][i - 1] if i > 0 else "Background"
        axes[0, i+1].set_title(title)
        axes[1, i+1].imshow(batch[3][0, i, :, :].cpu().detach().numpy(), cmap="binary")
        axes[1, i+1].axis("off")
        axes[1, i+1].set_title("Truth")
      plt.tight_layout()
      plt.savefig(os.path.join(self.img_path, "seg_P_"+str(patients_in_batch[0])+"_B_"+str(batch_idx)+".png"))
      plt.close()
      
    return {"patients_in_batch": patients_in_batch, "slices_in_batch": slices_in_batch, "seg_output": seg_output, "TAC_pred_batch": TAC_pred_batch}


  
  def on_test_epoch_end(self, outputs):
    run_name = os.path.split(os.path.split(self.pd_path)[0])[1]

    summary = dict()
    patient_totals = dict()
    patient_counts = dict()

    for o in outputs:
      metric_dict = o["metric_dict"]
      patients_in_batch = o["patients_in_batch"]
      slices_in_batch = o["slices_in_batch"]
      for i in range(len(patients_in_batch)):
        p = patients_in_batch[i]
        if not p in summary.keys(): 
          summary[p] = dict()
          patient_totals[p] = {"CosineSim": 0, "MSE": 0, "MAE": 0}
          patient_counts[p] = 0
        for j in range(len(slices_in_batch)):
          s = int(slices_in_batch[j].item())
          if patients_in_batch[j] == p:
            summary[p][s] = dict()
            summary[p][s]["MSE"] = metric_dict["mse"]
            summary[p][s]["MAE"] = metric_dict["mae"]
            summary[p][s]["CosineSim"] = metric_dict["cosine_sim"]

            patient_totals[p]["CosineSim"] += metric_dict["cosine_sim"]
            patient_totals[p]["MSE"] += metric_dict["mse"]
            patient_totals[p]["MAE"] += metric_dict["mae"]
            patient_counts[p] += 1

    patient_std_devs = {}
    for p in summary.keys():
      current_df = pd.DataFrame.from_dict(summary[p])
      # This file contains the metrics per slice. It allows to identify slices with bad peformance. 
      # It is also used during evaluation phase to compute the metrics on the whole dataset
      current_df.to_excel(os.path.join(self.pd_path, p + "_metric_per_slice_" + run_name + ".xlsx"))
      
      patient_metrics = pd.DataFrame.from_dict(summary[p]).transpose()
      patient_means = patient_metrics.mean()
      squared_diffs = (patient_metrics - patient_means) ** 2
      patient_std_devs[p] = np.sqrt(squared_diffs.mean()).to_dict()

    # Reconstruct the 3D kinetic parameters volumes
    reconstruct_prediction(self.pt_path, self.nifty_path)

    # Compute the average metrics per patient
    patient_averages = {p: {metric: total / patient_counts[p] for metric, total in totals.items()} for p, totals in patient_totals.items()}

    # Compute the average metrics over the whole dataset
    overall_averages = {metric: sum(patient_averages[p][metric] for p in patient_averages) / len(patient_averages) for metric in ["CosineSim", "MSE", "MAE"]}

    # Compute the standard deviation for the overall metrics
    overall_sums_of_squares = {"CosineSim": 0, "MSE": 0, "MAE": 0}
    for p in patient_averages.keys():
        for metric in ["CosineSim", "MSE", "MAE"]:
            overall_sums_of_squares[metric] += (patient_averages[p][metric] - overall_averages[metric]) ** 2

    overall_std_devs = {metric: np.sqrt(total / len(patient_averages)) for metric, total in overall_sums_of_squares.items()}

    print(f"Patient averages: {patient_averages}")
    print(f"Patient standard deviations: {patient_std_devs}")
    print(f"Overall standard deviations: {overall_std_devs}")
    print(f"Overall averages: {overall_averages}")

    return 

  def configure_optimizers(self):
    # Define the optimizer
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"], weight_decay=self.config['weight_decay'])#, betas=(self.config['beta1'], self.config['beta2']))
    
    # Define the learning rate scheduler
    if self.config["lr_scheduler"] == "CosineAnnealingLR":
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['cosine_T_max'], eta_min=self.config['cosine_eta_min'], last_epoch=-1),
            'interval': 'epoch',
            'frequency': 1,
        }

    elif self.config["lr_scheduler"] == "MultiStepLR":
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['multistep_milestones'], gamma=self.config['multistep_gamma'], last_epoch=-1),
            'interval': 'epoch',
            'frequency': 1,
        }

    elif self.config["lr_scheduler"] == "ReduceLROnPlateau":
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.config['reduce_factor'], patience=self.config['reduce_patience'], verbose=True),
            'monitor': self.config['reduce_monitor'],  # Specify the metric you want to monitor
            'interval': 'epoch',
            'frequency': 1 if self.config['reduce_monitor'] == "val_loss" else self.config['val_freq'],
            'strict': True,
        }

    return [optimizer], [scheduler]