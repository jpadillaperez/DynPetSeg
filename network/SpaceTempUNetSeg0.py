import os
import torch
import wandb
import numpy as np
import pandas as pd
import multiprocessing
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from monai.losses import DiceLoss

from dataset import DynPETDataset
from network.unet_blocks import UNet
from network.unet_blocks_ST import UNet_ST
import utils.similaritymeasures_torch as similaritymeasures_torch
from utils.utils_main import make_save_folder_struct, reconstruct_prediction
#----------------------------------------------

class SpaceTempUNetSeg0(pl.LightningModule):
  def __init__(self, config):
    # Enforce determinism
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    
    super(SpaceTempUNetSeg0, self).__init__()

    # Read configuration file and add new info if needed
    self.config = config
    self.config["output_size"] = len(self.config["segmentation_list"]) + 1

    print("\nConfiguration: ", self.config)

    if self.config["use_spatio_temporal_unet"]:
      self.model = UNet_ST(in_channels=1, out_channels=self.config["output_size"], config=self.config)
    else:
      self.model = UNet(in_channels=1, out_channels=self.config["output_size"], config=self.config)

    self.loss_function = DiceLoss(squared_pred=True, reduction="mean", softmax=True, include_background=True)
    #self.softmax = torch.nn.Softmax(dim=1)
    self.validation_step_outputs = []

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

      self.t = self.train_dataset.t.to(self.config["device"])
      self.time_stamp = self.train_dataset.time_stamp.to(self.config["device"])

      self.patch_size = self.train_dataset.__get_patch_size__()
      print("Patch size: ", self.patch_size)
      self.t_batch = self.t.repeat(self.patch_size*self.patch_size, 1, 1)
      self.time_stamp_batch = self.time_stamp.repeat(self.patch_size*self.patch_size, 1, 1)

    if stage == "test":
      self.test_dataset = DynPETDataset(self.config, "test")
      if self.config["remove_slices_without_segmentation"]:
        self.test_dataset.remove_slices_without_segmentation()
      self.patch_size = self.test_dataset.__get_patch_size__()
      print("Patch size: ", self.patch_size)
      self.idif_test_set = self.test_dataset.idif
      self.t = self.test_dataset.t.to(self.config["device"])
      self.time_stamp = self.test_dataset.time_stamp.to(self.config["device"])
      self.t_batch = self.t.repeat(self.patch_size*self.patch_size, 1, 1)
      self.time_stamp_batch = self.time_stamp.repeat(self.patch_size*self.patch_size, 1, 1)

  def forward(self, x):
    x = self.model(x)
    return x
  
  #def loss_function(self, pred, real):        
  #  loss = similaritymeasures_torch.dice(pred, real)
  #  return loss
  
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
    # batch = [patient, slice, TAC, mask]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1)  # adding channel dimension --> [b, 1, 62, w, h]
    x = F.pad(TAC_mes_batch, (0,0,0,0,1,1))   # padding --> [b, 1, 64, w, h]
    truth = F.one_hot(batch[3].long(), num_classes=len(self.config["segmentation_list"]) + 1).permute(0, 3, 1, 2).float()  # [b, SEG, w, h]

    # Forward pass
    output = self.forward(x)        # [b, SEG, 1, w, h]
    output = output.squeeze(2)  # [b, SEG, w, h]

    # Compute the loss with the segmentation mask
    #output = self.softmax(output)
    loss = self.loss_function(output, truth)

    output = torch.argmax(output, dim=1)
    truth = torch.argmax(truth, dim=1)

    # Gradient clipping
    if (self.config["clip_grad_norm"] > 0):
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["clip_grad_norm"])

    # Prepare data to log
    if batch_idx % self.config["log_img_freq"] == 0:   
      # Log Segmentation
      plt.figure()
      plt.suptitle("Patient: "+str(batch[0][0])+" Slice: "+str(batch[1][0]))
      #First subplot
      plt.subplot(1, 5, 1)
      plt.imshow(truth[0, :, :].cpu().detach().numpy(), cmap="gray")
      plt.axis("off")
      plt.title("Truth")
      plt.clim(0, len(self.config["segmentation_list"]))
      #Second subplot
      plt.subplot(1, 5, 2)
      plt.imshow(output[0, :, :].cpu().detach().numpy(), cmap="gray")
      plt.axis("off")
      plt.title("Output")
      plt.clim(0, len(self.config["segmentation_list"]))
      #Third subplot
      plt.subplot(1, 5, 3)
      plt.imshow(batch[2][0, 10, :, :].cpu().detach().numpy(), cmap="gray")
      plt.axis("off")
      plt.title("10t")
      plt.clim(0, np.max(batch[2][0, 10, :, :].cpu().detach().numpy()))
      #Fourth subplot
      plt.subplot(1, 5, 4)
      plt.imshow(batch[2][0, 20, :, :].cpu().detach().numpy(), cmap="gray")
      plt.axis("off")
      plt.title("20t")
      plt.clim(0, np.max(batch[2][0, 20, :, :].cpu().detach().numpy()))
      #Fifth subplot
      plt.subplot(1, 5, 5)
      plt.imshow(batch[2][0, 30, :, :].cpu().detach().numpy(), cmap="gray")
      plt.axis("off")
      plt.title("30t")
      plt.clim(0, np.max(batch[2][0, 30, :, :].cpu().detach().numpy()))
      wandb.log({"Segmentation (training batch)": wandb.Image(plt)})
      plt.close()

    self.log('train_loss', loss.item(), on_step=False, on_epoch=True)
    self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)

    return {"loss": loss}

  def validation_step(self, batch, batch_idx):

    # batch = [patient, slice, TAC, mask]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
    x = F.pad(TAC_mes_batch, (0,0,0,0,1,1), "replicate")   # padding --> [b, 1, 64, w, h]
    truth = F.one_hot(batch[3].long(), num_classes=len(self.config["segmentation_list"]) + 1).permute(0, 3, 1, 2).float()  # [b, SEG, w, h]

    output = self.forward(x)        # [b, SEG, 1, w, h]
    output = output.squeeze(2)

    # Compute the loss with the segmentation mask
    #output = self.softmax(output)
    loss = self.loss_function(output, truth)
    
    output = torch.argmax(output, dim=1).float()
    truth = torch.argmax(truth, dim=1).float()

    self.log('val_loss', loss.item(), on_step=False, on_epoch=True)

    # Prepare data to log
    if batch_idx % self.config["log_img_freq"] == 0:
      # Log Segmentation
      fig = plt.figure()
      plt.suptitle("Patient: "+str(batch[0][0])+" Slice: "+str(batch[1][0]))
      #First subplot
      plt.subplot(1, 5, 1)
      plt.imshow(truth[0, :, :].cpu().detach().numpy(), cmap="gray")
      plt.axis("off")
      plt.title("Truth")
      plt.clim(0, len(self.config["segmentation_list"]))
      #Second subplot
      plt.subplot(1, 5, 2)
      plt.imshow(output[0, :, :].cpu().detach().numpy(), cmap="gray")
      plt.axis("off")
      plt.title("Output")
      plt.clim(0, len(self.config["segmentation_list"]))
      # Third subplot
      plt.subplot(1, 5, 3)
      plt.imshow(batch[2][0, 10, :, :].cpu().detach().numpy(), cmap="gray")
      plt.axis("off")
      plt.title("10t")
      plt.clim(0, np.max(batch[2][0, 10, :, :].cpu().detach().numpy()))
      # Fourth subplot
      plt.subplot(1, 5, 4)
      plt.imshow(batch[2][0, 20, :, :].cpu().detach().numpy(), cmap="gray")
      plt.axis("off")
      plt.title("20t")
      plt.clim(0, np.max(batch[2][0, 20, :, :].cpu().detach().numpy()))
      # Fifth subplot
      plt.subplot(1, 5, 5)
      plt.imshow(batch[2][0, 30, :, :].cpu().detach().numpy(), cmap="gray")
      plt.axis("off")
      plt.title("30t")
      plt.clim(0, np.max(batch[2][0, 30, :, :].cpu().detach().numpy()))
      fig_seg = {"Segmentation (validation batch)": wandb.Image(fig)}
      plt.close()

      self.validation_step_outputs.append({"fig_seg": fig_seg})
      return {"val_loss": loss.item(), "fig_seg": fig_seg}

    else:
      self.validation_step_outputs.append({"val_loss": loss.item(), "fig_seg": None})
      return {'val_loss': loss.item(), "fig_seg": None}
  
  def on_validation_epoch_end(self):
    for o in self.validation_step_outputs:
      if not o["fig_seg"] is None:  wandb.log(o["fig_seg"])
    self.validation_step_outputs.clear()
    return
  
  def test_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC, mask]
    patients_in_batch = batch[0]
    slices_in_batch = batch[1]
    mask = batch[3]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
    x = F.pad(TAC_mes_batch, (0,0,0,0,1,1), "replicate")   # padding --> [b, 1, 64, w, h]

    logits_params = self.forward(x)        # [b, 4, 1, w, h]

    # Compute the loss with the segmentation mask
    loss = self.loss_function(logits_params, mask)

    # Save predictions
    current_run_name = wandb.run.name
    resume_run_name = os.path.split(os.path.split(self.trainer.ckpt_path)[0])[1]
    self.img_path, self.pd_path, self.pt_path, self.nifty_path = make_save_folder_struct(current_run_name, resume_run_name, root_checkpoints_path, self.trainer.ckpt_path)
    to_save = [patients_in_batch, slices_in_batch, mask]
    s = int(slices_in_batch.item())
    torch.save(to_save, os.path.join(self.pt_path, "P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".pt"))

    # Prepare data to log
    if batch_idx % 50 == 0:
      if not len(slices_in_batch) == 1: s = slices_in_batch

      # Log slices
      fig = plt.figure()
      plt.imshow(batch[1].cpu().detach().numpy(), cmap="gray")
      fig_slice = {"Slice (test batch: "+str(batch_idx)+")": wandb.Image(fig)}
      plt.savefig(os.path.join(self.img_path, "slice_P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".png"))
      plt.close()

      # Log Segmentation
      fig = plt.figure()
      plt.imshow(mask.cpu().detach().numpy(), cmap="gray")
      fig_seg = {"Segmentation (test batch: "+str(batch_idx)+")": wandb.Image(fig)}
      plt.savefig(os.path.join(self.img_path, "seg_P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".png"))
      plt.close()


    return {"patients_in_batch": patients_in_batch, "slices_in_batch": slices_in_batch}#, "metric_dict": metric_dict}
  
  #def on_test_epoch_end(self, outputs):
  #  run_name = os.path.split(os.path.split(self.pd_path)[0])[1]
#
  #  summary = dict()
  #  patient_totals = dict()
  #  patient_counts = dict()
#
  #  for o in outputs:
  #    metric_dict = o["metric_dict"]
  #    patients_in_batch = o["patients_in_batch"]
  #    slices_in_batch = o["slices_in_batch"]
  #    for i in range(len(patients_in_batch)):
  #      p = patients_in_batch[i]
  #      if not p in summary.keys(): 
  #        summary[p] = dict()
  #        patient_totals[p] = {"CosineSim": 0, "MSE": 0, "MAE": 0}
  #        patient_counts[p] = 0
  #      for j in range(len(slices_in_batch)):
  #        s = int(slices_in_batch[j].item())
  #        if patients_in_batch[j] == p:
  #          summary[p][s] = dict()
  #          summary[p][s]["MSE"] = metric_dict["mse"]
  #          summary[p][s]["MAE"] = metric_dict["mae"]
  #          summary[p][s]["CosineSim"] = metric_dict["cosine_sim"]
#
  #          patient_totals[p]["CosineSim"] += metric_dict["cosine_sim"]
  #          patient_totals[p]["MSE"] += metric_dict["mse"]
  #          patient_totals[p]["MAE"] += metric_dict["mae"]
  #          patient_counts[p] += 1
#
  #  patient_std_devs = {}
  #  for p in summary.keys():
  #    current_df = pd.DataFrame.from_dict(summary[p])
  #    # This file contains the metrics per slice. It allows to identify slices with bad peformance. 
  #    # It is also used during evaluation phase to compute the metrics on the whole dataset
  #    current_df.to_excel(os.path.join(self.pd_path, p + "_metric_per_slice_" + run_name + ".xlsx"))
  #    
  #    patient_metrics = pd.DataFrame.from_dict(summary[p]).transpose()
  #    patient_means = patient_metrics.mean()
  #    squared_diffs = (patient_metrics - patient_means) ** 2
  #    patient_std_devs[p] = np.sqrt(squared_diffs.mean()).to_dict()
#
  #  # Reconstruct the 3D kinetic parameters volumes
  #  reconstruct_prediction(self.pt_path, self.nifty_path)
#
  #  # Compute the average metrics per patient
  #  patient_averages = {p: {metric: total / patient_counts[p] for metric, total in totals.items()} for p, totals in patient_totals.items()}
#
  #  # Compute the average metrics over the whole dataset
  #  overall_averages = {metric: sum(patient_averages[p][metric] for p in patient_averages) / len(patient_averages) for metric in ["CosineSim", "MSE", "MAE"]}
#
  #  # Compute the standard deviation for the overall metrics
  #  overall_sums_of_squares = {"CosineSim": 0, "MSE": 0, "MAE": 0}
  #  for p in patient_averages.keys():
  #      for metric in ["CosineSim", "MSE", "MAE"]:
  #          overall_sums_of_squares[metric] += (patient_averages[p][metric] - overall_averages[metric]) ** 2
#
  #  overall_std_devs = {metric: np.sqrt(total / len(patient_averages)) for metric, total in overall_sums_of_squares.items()}
#
  #  print(f"Patient averages: {patient_averages}")
  #  print(f"Patient standard deviations: {patient_std_devs}")
  #  print(f"Overall standard deviations: {overall_std_devs}")
  #  print(f"Overall averages: {overall_averages}")
#
  #  return 

  def configure_optimizers(self):
    # Define the optimizer
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"], weight_decay=self.config['weight_decay'], betas=(self.config['beta1'], self.config['beta2']))
    
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