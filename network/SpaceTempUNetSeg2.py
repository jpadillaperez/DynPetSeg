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
from network.unet_blocks import UNet
from network.unet_blocks_ST import UNet_ST
from network.unet_blocks_2D import UNet_2D
from utils.utils_kinetic import PET_2TC_KM_batch
from utils.utils_logging import log_slice, log_curves, mask_data
import utils.similaritymeasures_torch as similaritymeasures_torch
from utils.utils_main import make_save_folder_struct, reconstruct_prediction, apply_final_activation
from utils.utils_torch import torch_interp_Nd, WarmupScheduler, weights_init_kaiming, weights_init_xavier
#----------------------------------------------

class SpaceTempUNetSeg2(pl.LightningModule):
  def __init__(self, config):
    # Enforce determinism
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    
    super(SpaceTempUNetSeg2, self).__init__()

    # Read configuration file and add new info if needed
    self.config = config
    self.config["output_size"] = 4  # This is the number of output channels in kinetic network (1 per kinetic parameter)
    self.config["output_size_seg"] = len(self.config["segmentation_list"]) + 1  # This is the number of output channels after second network (1 per segmentation class + background)
    self.config["mask_loss"] = True
    self.config["multi_clamp_params"] = {"k1": (0.01, 2), "k2": (0.01, 3), "k3": (0.01, 1), "Vb": (0, 1)}
    self.config["use_pearson_metric"] = False     # Setting it to True may slow down the training 

    # The paper was developed with config["patch_size"] = 112. However in this was some regions of the body are outsie the FOV (for example the arms of part of the belly or the back).
    # While config["patch_size"] = 112 doesn't reduce the performance of the training, using a larger patch_size will allow to infere complete 3D volumes without missing parts. 
    
    print("\nConfiguration: ", self.config)

    # Kinetic network
    if self.config["use_spatio_temporal_unet"]:
      self.model_kinetics = UNet_ST(in_channels=1, out_channels=self.config["output_size"], config=self.config)
    else:
      self.model_kinetics = UNet(in_channels=1, out_channels=self.config["output_size"], config=self.config)

    # Segmentation network
    self.model_segmentation = UNet_2D(in_channels=self.config["output_size"], out_channels=self.config["output_size_seg"], config=self.config)

    # Initialize weights
    if self.config["weight_init"] == "kaiming":
      self.model_kinetics.apply(weights_init_kaiming)
    elif self.config["weight_init"] == "xavier":
      self.model_kinetics.apply(weights_init_xavier)

    # Loss function
    self.dice_loss = DiceLoss(squared_pred=True, reduction="mean", softmax=True, include_background=True)
    self.softmax = torch.nn.Softmax(dim=1)

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

    
  def forward_kinetics(self, x):
    x = self.model_kinetics(x)
    return x
  
  def forward_segmentation(self, x):
    x = self.model_segmentation(x)
    return x

  
  def loss_function(self, pred_TAC, real_TAC):
    loss = similaritymeasures_torch.mse(pred_TAC.to(self.config["device"]).double(), real_TAC.to(self.config["device"]).double())
    return loss

  def seg_loss_function(self, pred_seg, real_seg, real_TAC):
    if self.config["mask_loss"]:
      time_stamp_batch = self.time_stamp_batch[:, 0, :].permute((1, 0))
      b, t, h, w = real_TAC.shape                         # [b, 62, w, h]
      real_TAC = torch.reshape(real_TAC, [b, t, h*w])     # [b, 62, w*h]
      real_seg = torch.reshape(real_seg, [b, len(self.config["segmentation_list"]) + 1, h*w])  # [b, Seg, w*h]
      pred_seg = torch.reshape(pred_seg, [b, len(self.config["segmentation_list"]) + 1, h*w])  # [b, Seg, w*h]
      seg_loss = 0
      for i in range(b):
        current_TAC_batch = real_TAC[i, :, :]
        AUC = torch.trapezoid(current_TAC_batch, time_stamp_batch, dim=0)
        maskk = AUC > 10
        maskk = maskk * 1
        mask = maskk.repeat(len(self.config["segmentation_list"]) + 1, 1)
        pred_seg = torch.multiply(pred_seg, mask)
        real_seg = torch.multiply(real_seg, mask)
        seg_loss += self.dice_loss(pred_seg, real_seg)
      return seg_loss/b
    else:
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
    
  def accumulate_func(self, batch, logits, return_value=None):
      p = batch[0]              # len(p) = b
      TAC_batch = batch[2]      # [b, 62, w, h]

      b, t, h, w = TAC_batch.shape
      TAC_batch = torch.reshape(TAC_batch, [b, t, h*w])               # measurements
      TAC_pred_batch = torch.zeros_like(TAC_batch)

      kinetic_params = apply_final_activation(logits, self.config)          #  [b, 4, 1, 112, 112]
      kinetic_params = torch.reshape(kinetic_params, [b, 4, h*w, 1])   # [b, 4, w*h, 1]
      kinetic_params = kinetic_params.repeat(1, 1, 1, len(self.t))  #  [b, 4, w*h, 600]

      logits = torch.reshape(logits, [b, 4, h*w, 1])   # predicted params [b, 4, 1, 112, 112] --> [b, 4, w*h, 1]
      
      loss = 0

      metric_mse = 0
      metric_mae = 0
      metric_abc = 0
      cosine_sim = 0
      r2 = 0
      chi2 = 0
      pearson = 0
      pearson_p = 0
      counts_pearson = 0

      if not h*w == self.patch_size*self.patch_size:
        print("ERROR: the wrong patch size is used!")
        return

      time_stamp_batch = self.time_stamp_batch[:, 0, :].permute((1, 0))
      for i in range(b):
        current_TAC_batch = TAC_batch[i, :, :]
        current_kinetic_params = kinetic_params[i, :, :, :]
        
        current_TAC_pred_batch_long, _ = self.make_curve(current_kinetic_params, p[i])
        
        # Fix the mismatch in time: input has 62 time points, model has 600
        current_TAC_pred_batch_long = current_TAC_pred_batch_long[0, :, :].permute((1, 0))
        current_TAC_pred_batch = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], current_TAC_pred_batch_long)
        current_TAC_pred_batch = current_TAC_pred_batch.permute((1, 0))

        if self.config["mask_loss"]:    # More details about this can be found in utils/utils_logging.py
          AUC = torch.trapezoid(current_TAC_batch, time_stamp_batch, dim=0)
          maskk = AUC > 10
          maskk = maskk * 1
          mask = maskk.repeat(62, 1)
          current_TAC_batch = torch.multiply(current_TAC_batch, mask)
          current_TAC_pred_batch = torch.multiply(current_TAC_pred_batch, mask)
        
        TAC_pred_batch[i, :, :] = current_TAC_pred_batch
        if return_value is None or return_value == "Loss":  
          loss += self.loss_function(current_TAC_pred_batch, current_TAC_batch)

        if return_value is None or return_value == "Metric":  
          if self.config["mask_loss"]:
              square = torch.square(current_TAC_pred_batch.to(self.config["device"]) - current_TAC_batch.to(self.config["device"]))
              metric_mse += torch.sum(square).item() / len(mask[mask>0])

              absolute = torch.abs(current_TAC_pred_batch.to(self.config["device"]) - current_TAC_batch.to(self.config["device"]))
              metric_mae += torch.sum(absolute).item() / len(mask[mask>0])

              cosine_sim_slice = torch.nn.functional.cosine_similarity(current_TAC_pred_batch.to(self.config["device"]), current_TAC_batch.to(self.config["device"]), 0)
              cosine_sim += torch.sum(cosine_sim_slice).item() / len(maskk[maskk>0])

              weights = (self.frame_duration_batch * torch.exp(-0.00631 * time_stamp_batch)) / (current_TAC_batch.to(self.config["device"]))
              weights = torch.nan_to_num(weights, posinf=1)
              chi2_slice = torch.sum(torch.multiply(square, weights), axis=0)
              chi2 += torch.sum(chi2_slice).item() / len(maskk[maskk>0])
          else:
              metric_mse += similaritymeasures_torch.mse(current_TAC_pred_batch.to(self.config["device"]), current_TAC_batch.to(self.config["device"])).item()
              metric_mae += similaritymeasures_torch.mae(current_TAC_pred_batch.to(self.config["device"]), current_TAC_batch.to(self.config["device"])).item()
              cosine_sim += torch.mean(torch.nn.functional.cosine_similarity(current_TAC_pred_batch.to(self.config["device"]), current_TAC_batch.to(self.config["device"]), 0)).item()

              square = torch.square(current_TAC_pred_batch.to(self.config["device"]) - current_TAC_batch.to(self.config["device"]))
              weights = (self.frame_duration_batch * torch.exp(-0.00631 * time_stamp_batch)) / (current_TAC_batch.to(self.config["device"]))
              weights = torch.nan_to_num(weights, posinf=1)
              chi2_slice = torch.sum(torch.multiply(square, weights), axis=0)
              chi2 += torch.mean(chi2_slice).item()
          
          # The following metrics are the same independently from self.config["mask_loss"]
          r2 += similaritymeasures_torch.r2(current_TAC_pred_batch.to(self.config["device"]), current_TAC_batch.to(self.config["device"])).item()

          if self.config["use_pearson_metric"]:
              for j in range(h*w):
                  current_pred_TAC = current_TAC_pred_batch[:, j]
                  current_real_TAC = current_TAC_batch[:, j]
                  if self.config["mask_loss"]: 
                      if torch.sum(current_pred_TAC) == 0 or torch.sum(current_real_TAC) == 0:
                          continue
                      else:
                          current_pearson, current_p = pearsonr(current_pred_TAC.cpu().detach().numpy(), current_real_TAC.cpu().detach().numpy())
                          pearson += current_pearson
                          pearson_p += current_p
                          counts_pearson += 1

      loss = loss/b           # Use b instead od self.config["batch_size"] to accomodate for batches of different size (like the last one)
      loss_dict =  {"loss": loss}

      metric_mse = metric_mse/b
      metric_mae = metric_mae/b
      metric_abc = metric_abc/b
      cosine_sim = cosine_sim/b
      r2 = r2/b
      if self.config["use_pearson_metric"]:
        pearson_p = pearson_p/counts_pearson
        pearson = pearson/counts_pearson
      
      metric_dict = {"mse": metric_mse, "mae": metric_mae, "cosine_sim": cosine_sim, "r2": r2}
      if self.config["use_pearson_metric"]:
        metric_dict["pearson_corr"] = pearson
        metric_dict["pearson_p"] = pearson_p
      
      TAC_pred_batch = torch.reshape(TAC_pred_batch, [b, 1, t, h, w])

      if return_value is None:
        return loss_dict, metric_dict, TAC_pred_batch
      elif return_value == "Loss":
        return loss_dict
      elif return_value == "Metric":
        return metric_dict, TAC_pred_batch

  def accumulate_loss(self, batch, logits):
    return self.accumulate_func(batch, logits, return_value="Loss")
  
  def accumulate_metric(self, batch, logits):
    return self.accumulate_func(batch, logits, return_value="Metric")
  
  def accumulate_loss_and_metric(self, batch, logits):
    return self.accumulate_func(batch, logits, return_value=None)

  
  def make_curve(self, kinetic_params, patient):

    # Prepare the IDIF --> can't be moved outside the for loop because the patient can change inside the batch
    if self.stage == "fit" and patient in self.idif_train_set.keys():
      current_idif = self.idif_train_set[patient]
    elif self.stage == "fit" and patient in self.idif_val_set.keys():
      current_idif = self.idif_val_set[patient]
    elif self.stage == "test" and patient in self.idif_test_set.keys():
      current_idif = self.idif_test_set[patient]
    else: 
      print("ERROR: IDIF of patient "+str(patient)+" not found !")
      return
    idif_batch = current_idif.repeat(1, self.patch_size*self.patch_size, 1)

    # Prepare the kinetic parameters
    kinetic_params = kinetic_params.permute((1, 0, 2))
    k1 = kinetic_params[:, 0, :].unsqueeze(1)
    k2 = kinetic_params[:, 1, :].unsqueeze(1)        
    k3 = kinetic_params[:, 2, :].unsqueeze(1)
    Vb = kinetic_params[:, 3, :].unsqueeze(1)

    # Compute the TAC
    current_pred_curve = PET_2TC_KM_batch(idif_batch.to(self.config["device"]), self.t_batch.to(self.config["device"]), k1.to(self.config["device"]), k2.to(self.config["device"]), k3.to(self.config["device"]), Vb.to(self.config["device"]))

    return current_pred_curve, None



  def training_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1)  # adding channel dimension --> [b, 1, 62, w, h]
    x = F.pad(TAC_mes_batch, (0,0,0,0,1,1))   # padding --> [b, 1, 64, w, h]
    truth_seg = F.one_hot(batch[3].long(), num_classes= len(self.config["segmentation_list"]) + 1).permute(0, 3, 1, 2).float()  # [b, SEG, w, h]

    #Kinetic network
    logits_params = self.forward_kinetics(x)        # [b, 4, 1, w, h]
    _, _, TAC_pred_batch = self.accumulate_loss_and_metric(batch=batch, logits=logits_params)
    if self.config["mask_loss"]:
      _, _, logits_params = mask_data(TAC_mes_batch, TAC_pred_batch, logits_params, self.time_stamp, patch_size=self.patch_size)

    # Segmentation network
    logits_seg = self.forward_segmentation(logits_params[:, :, 0, :, :])        # [b, Seg, w, h]

    # Segmentation Loss
    loss = self.seg_loss_function(logits_seg, truth_seg, batch[2])

    # Backward pass
    self.log('train_loss', loss, on_step=False, on_epoch=True)
    self.log('Learning_Rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #Print the max gradient
    max_grad = 0
    for name, param in self.model_kinetics.named_parameters():
      if param.requires_grad:
        if param.grad is not None:
          max_grad = max(max_grad, param.grad.abs().max().item())
    
    self.log('max_grad_kinetics', max_grad, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    max_grad = 0
    for name, param in self.model_segmentation.named_parameters():
      if param.requires_grad:
        if param.grad is not None:
          max_grad = max(max_grad, param.grad.abs().max().item())
    
    self.log('max_grad_segmentation', max_grad, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    if ((np.random.rand() < 0.5) and (self.num_train_log_images < self.config["log_train_imgs"])):
      # ------------------ Log Slice ------------------
      kinetic_params = apply_final_activation(logits_params, self.config)
      fig = log_slice(self.config, TAC_mes_batch, kinetic_params)
      wandb.log({"Slice (training batch)": wandb.Image(fig)})
      plt.close()
      
      #------------------ Log Segmentation ------------------
      output_seg = logits_seg
      fig, axes = plt.subplots(2, len(self.config["segmentation_list"]) + 2, figsize=((len(self.config["segmentation_list"]) + 2) * 2, len(self.config["segmentation_list"]) + 2))
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
      for i in range(0, (len(self.config["segmentation_list"]) + 1), 1):
        axes[0, i+1].imshow(output_seg[0, i, :, :].cpu().detach().numpy(), cmap="binary")
        axes[0, i+1].axis("off")
        title = self.config["segmentation_list"][i - 1] if i > 0 else "Background"
        axes[0, i+1].set_title(title)
        axes[1, i+1].imshow(truth_seg[0, i, :, :].cpu().detach().numpy(), cmap="binary")
        axes[1, i+1].axis("off")
        axes[1, i+1].set_title("Truth")
      plt.tight_layout()
      wandb.log({"Segmentation (training batch)": wandb.Image(fig)})
      plt.close()

      self.num_train_log_images += 1

    return {"loss": loss}

  def on_train_epoch_end(self, outputs):
    self.num_train_log_images = 0
    return 

  def validation_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
    x = torch.nn.functional.pad(TAC_mes_batch, (0,0,0,0,1,1), "replicate")   # padding --> [b, 1, 64, w, h]
    truth_seg = F.one_hot(batch[3].long(), num_classes= len(self.config["segmentation_list"]) + 1).permute(0, 3, 1, 2).float()  # [b, SEG, w, h]

    # Kinetic network
    logits_params = self.forward_kinetics(x)        # [b, 4, 1, w, h]
    _, _, TAC_pred_batch = self.accumulate_loss_and_metric(batch=batch, logits=logits_params)
    if self.config["mask_loss"]:
      _, _, logits_params = mask_data(TAC_mes_batch, TAC_pred_batch, logits_params, self.time_stamp, patch_size=self.patch_size)

    # Segmentation network
    logits_seg = self.forward_segmentation(logits_params[:, :, 0, :, :])        # [b, Seg, w, h]

    # Segmentation Loss
    loss = self.seg_loss_function(logits_seg, truth_seg, batch[2])

    # Log the loss
    self.log('val_loss', loss, on_step=False, on_epoch=True)

    # Prepare data to log
    if ((np.random.rand() < 0.5) and (self.num_val_log_images < self.config["log_val_imgs"])):
      # Log slices
      kinetic_params = apply_final_activation(logits_params, self.config)
      fig = log_slice(self.config, TAC_mes_batch, kinetic_params)
      fig_slice = {"Slice (validation batch)": wandb.Image(fig)}
      plt.close()

      # ------------------ Log Segmentation ------------------
      output_seg = logits_seg
      fig, axes = plt.subplots(2, len(self.config["segmentation_list"]) + 2, figsize=((len(self.config["segmentation_list"]) + 2) * 2, len(self.config["segmentation_list"]) + 2))
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
      for i in range(0, (len(self.config["segmentation_list"]) + 1), 1):
        axes[0, i+1].imshow(output_seg[0, i, :, :].cpu().detach().numpy(), cmap="binary")
        axes[0, i+1].axis("off")
        title = self.config["segmentation_list"][i - 1] if i > 0 else "Background"
        axes[0, i+1].set_title(title)
        axes[1, i+1].imshow(truth_seg[0, i, :, :].cpu().detach().numpy(), cmap="binary")
        axes[1, i+1].axis("off")
        axes[1, i+1].set_title("Truth")
      plt.tight_layout()
      fig_seg = {"Segmentation (validation batch)": wandb.Image(plt)}
      plt.close()

      self.num_val_log_images += 1

      self.validation_step_outputs.append({"fig_seg": fig_seg, "fig_slice": fig_slice})
      return {'val_loss': loss, "fig_seg": fig_seg, "fig_slice": fig_slice}
    else:
      self.validation_step_outputs.append({"val_loss": loss, "fig_seg": None, "fig_slice": None})
      return {'val_loss': loss, "fig_seg": None, "fig_slice": None}
  
  def on_validation_epoch_end(self):
    for o in self.validation_step_outputs:
      if not o["fig_seg"] is None:
        wandb.log(o["fig_seg"])
      if not o["fig_slice"] is None:
        wandb.log(o["fig_slice"])
    
    self.validation_step_outputs.clear()
    self.num_val_log_images = 0
    return
  
  def test_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    patients_in_batch = batch[0]
    slices_in_batch = batch[1]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
    x = torch.nn.functional.pad(TAC_mes_batch, (0,0,0,0,1,1), "replicate")   # padding --> [b, 1, 64, w, h]

    logits_params = self.forward_kinetics(x)        # [b, 4, 1, w, h]
    kinetic_params = apply_final_activation(logits_params, self.config)
    logits_seg = self.forward_segmentation(x)        # [b, 4, 1, w, h]
    seg = self.softmax(logits_seg)

    # Save predictions
    current_run_name = wandb.run.name
    resume_run_name = os.path.split(os.path.split(self.trainer.ckpt_path)[0])[1]
    self.img_path, self.pd_path, self.pt_path, self.nifty_path = make_save_folder_struct(current_run_name, resume_run_name, root_checkpoints_path, self.trainer.ckpt_path)
    to_save = [patients_in_batch, slices_in_batch, seg]
    s = int(slices_in_batch.item())
    torch.save(to_save, os.path.join(self.pt_path, "P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".pt"))

    # Prepare data to log
    if batch_idx % 50 == 0:
      if not len(slices_in_batch) == 1: s = slices_in_batch

      # Log TAC             
      log_curves(TAC_mes_batch[:, :, 0:62, :, :].cpu().detach().numpy(), TAC_pred_batch.cpu().detach().numpy(), self.time_stamp.to("cpu"), self.time_stamp.to("cpu"), self.current_epoch)
      plt.savefig(os.path.join(self.img_path, "TAC_P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".png"))
      plt.close()

      # Log slices
      log_slice(self.config, TAC_mes_batch, kinetic_params)
      plt.savefig(os.path.join(self.img_path, "slices_P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".png"))
      plt.close()

    return {"patients_in_batch": patients_in_batch, "slices_in_batch": slices_in_batch, "metric_dict": metric_dict}
  
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

    # Define the warmup scheduler
    #warmup_scheduler = WarmupScheduler(optimizer, total_epochs=self.config["epochs"], warmup_epochs=self.config["warmup_epochs"])

    return [optimizer], [scheduler]