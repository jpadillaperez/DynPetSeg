import os
import torch
import wandb
import json
import numpy as np
import pandas as pd
import multiprocessing
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import pearsonr
from monai.losses import DiceLoss, DiceFocalLoss

from dataset import DynPETDataset
from network.unet_blocks_ST_2Heads import UNet_ST_2Heads
from utils.utils_kinetic import PET_2TC_KM_batch
from utils.utils_logging import log_slice, log_curves, mask_data
from utils.set_root_paths import root_path, root_checkpoints_path
import utils.similaritymeasures_torch as similaritymeasures_torch
from utils.utils_main import make_save_folder_struct, reconstruct_prediction, apply_final_activation
from utils.utils_torch import torch_interp_Nd, WarmupScheduler, weights_init_kaiming, weights_init_xavier
#----------------------------------------------

class BaseTrainingModule(pl.LightningModule):
  def __init__(self, config):
    # Enforce determinism
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    
    super(BaseTrainingModule, self).__init__()

    # Read configuration file and add new info if needed
    self.config = config
    self.config["output_size"] = 4  # This is the number of output channels in kinetic network (1 per kinetic parameter)
    self.config["output_size_seg"] = len(self.config["segmentation_list"]) + 1  # This is the number of output channels after second network (1 per segmentation class + background)
    self.config["multi_clamp_params"] = {"k1": (0.01, 2), "k2": (0.01, 3), "k3": (0.01, 1), "Vb": (0, 1)}

    print("\nConfiguration: ", self.config)

    # Loss function
    self.dice_loss = DiceLoss(reduction="mean", softmax=True, include_background=True)
    self.dice_focal_loss = DiceFocalLoss(sigmoid=False, softmax=True, gamma=0.5, weight=None, squared_pred=False, reduction="mean")
    self.softmax = torch.nn.Softmax(dim=1) #Just for the test_step

    frame_duration = [10, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 30, 30, 60, 60, 60, 60, 120, 120, 120, 120, 120, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    self.frame_duration = np.array(frame_duration) / 60  # from s to min

    self.num_val_log_images = 0
    self.num_train_log_images = 0


  def initialize_weights(self, model):
    # Initialize weights
    if self.config["init_weights"] == "kaiming":
      model.apply(weights_init_kaiming)
    elif self.config["init_weights"] == "xavier":
      model.apply(weights_init_xavier)
    else:
      print("ERROR: weight initialization not recognized")
      return


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
      self.t = self.train_dataset.t.to(self.device)
      self.time_stamp = self.train_dataset.time_stamp.to(self.device)
      self.frame_duration_batch = torch.from_numpy(np.array(self.frame_duration)).unsqueeze(-1).repeat(1, self.patch_size*self.patch_size).to(self.device)
      
      self.t_batch = self.t.repeat(self.patch_size*self.patch_size, 1, 1)
      self.time_stamp_batch = self.time_stamp.repeat(self.patch_size*self.patch_size, 1, 1)

    if stage == "test":
      self.test_dataset = DynPETDataset(self.config, "test", patch_size=self.config["patch_size"])
      if self.config["remove_slices_without_segmentation"]:
        self.test_dataset.remove_slices_without_segmentation()

      self.patch_size = self.test_dataset.__get_patch_size__()
      self.idif_test_set = self.test_dataset.idif
      self.t = self.test_dataset.t.to(self.device)
      self.time_stamp = self.test_dataset.time_stamp.to(self.device)
      self.frame_duration_batch = torch.from_numpy(np.array(self.frame_duration)).unsqueeze(-1).repeat(1, self.patch_size*self.patch_size).to(self.device)
      
      self.t_batch = self.t.repeat(self.patch_size*self.patch_size, 1, 1)
      self.time_stamp_batch = self.time_stamp.repeat(self.patch_size*self.patch_size, 1, 1)

      self.summary = dict()
      self.patient_counts = dict()
      self.patient_totals = dict()


    self.load_from_checkpoint(kinetic_ckpt=self.config["kinetic_resume_path"], full_ckpt=self.config["full_resume_path"])

    self.max_grad_kin = 0
    self.max_grad_seg = 0

  def kin_loss_function(self, pred_TAC, real_TAC):
    return similaritymeasures_torch.mse(pred_TAC.to(self.device).double(), real_TAC.to(self.device).double())

  def seg_loss_function(self, pred_seg, real_seg):
    if self.config["loss_function"] == "dice":
      return self.dice_loss(pred_seg, real_seg)
    elif self.config["loss_function"] == "dice_focal":
      return self.dice_focal_loss(pred_seg, real_seg)
    else:
      print("ERROR: loss function not recognized")
      return

  def loss_combination(self, loss_kin, loss_seg):
    if self.config["loss_combination"] == "sum":
      return loss_kin + loss_seg
    elif self.config["loss_combination"] == "weighted_sum":
      return self.config["rescaling_factor_kinetic_loss"] * loss_kin + loss_seg
    else:
      print("ERROR: loss combination not recognized")
      return

  def train_dataloader(self):
    if self.config["num_devices"] > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=self.config["num_devices"], rank=self.config["rank"])
      train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config["batch_size"], sampler=train_sampler, num_workers=0)
    else:
      train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0)
    return train_loader

  def val_dataloader(self):
    if self.config["num_devices"] > 1:
      val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, num_replicas=self.config["num_devices"], rank=self.config["rank"])
      val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config["batch_size"], sampler=val_sampler, num_workers=0)
    else:
      val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=0)
    return val_loader
  
  def test_dataloader(self):
    test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
    return test_loader
    
  def accumulate_func(self, batch, logits, return_value=None):
    # batch = [patient, slice, TAC, seg]
    # logits = [b, 4, 1, 192, 192]
    p = batch[0]
    TAC_batch = batch[2]      # [b, 62, w, h]
    b, t, h, w = TAC_batch.shape
    TAC_batch = torch.reshape(TAC_batch, [b, t, h*w])    #[b, 62, 192*192]
    TAC_pred_batch = torch.zeros_like(TAC_batch)        #[b, 62, 192*192] 
    kinetic_params = apply_final_activation(logits, self.config)          #  [b, 4, 1, 192, 192]
    kinetic_params = torch.reshape(kinetic_params, [b, 4, h*w, 1])   # [b, 4, 192*192, 1]
    kinetic_params = kinetic_params.repeat(1, 1, 1, len(self.t))  #  [b, 4, w*h, 600]
    logits = torch.reshape(logits, [b, 4, h*w, 1]).to(self.device)   # predicted params [b, 4, 1, 112, 112] --> [b, 4, w*h, 1] [8, 4, 192*192, 1]
    
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

    if not h*w == self.patch_size*self.patch_size: # 192*192
      print("ERROR: the wrong patch size is used!")
      return

    time_stamp_batch = self.time_stamp_batch[:, 0, :].permute((1, 0)).to(self.device)
    for i in range(b):
      current_TAC_batch = TAC_batch[i, :, :].to(self.device) # [62, 192*192]
      current_kinetic_params = kinetic_params[i, :, :, :] # [4, 192*192, 600]
      current_TAC_pred_batch_long, _ = self.make_curve(current_kinetic_params, p[i])
      
      # Fix the mismatch in time: input has 62 time points, model has 600
      current_TAC_pred_batch_long = current_TAC_pred_batch_long[0, :, :].permute((1, 0))
      current_TAC_pred_batch = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], current_TAC_pred_batch_long).to(self.device)
      current_TAC_pred_batch = current_TAC_pred_batch.permute((1, 0))
      
      if self.config["mask_loss"]:    # More details about this can be found in utils/utils_logging.py
        AUC = torch.trapezoid(current_TAC_batch, time_stamp_batch, dim=0).to(self.device)
        maskk = AUC > 10
        maskk = maskk * 1
        mask = maskk.repeat(62, 1)
        current_TAC_batch = torch.multiply(current_TAC_batch, mask)
        current_TAC_pred_batch = torch.multiply(current_TAC_pred_batch, mask).to(self.device)
      
      TAC_pred_batch[i, :, :] = current_TAC_pred_batch
      if return_value is None or return_value == "Loss":  
        loss += self.kin_loss_function(current_TAC_pred_batch, current_TAC_batch)
        #loss += similaritymeasures_torch.mse(current_TAC_pred_batch.to(self.device).double(), current_TAC_batch.to(self.device).double())
      if return_value is None or return_value == "Metric":  
        if self.config["mask_loss"]:
          square = torch.square(current_TAC_pred_batch.to(self.device) - current_TAC_batch.to(self.device))
          metric_mse += torch.sum(square).item() / len(mask[mask>0])

          absolute = torch.abs(current_TAC_pred_batch.to(self.device) - current_TAC_batch.to(self.device))
          metric_mae += torch.sum(absolute).item() / len(mask[mask>0])

          cosine_sim_slice = torch.nn.functional.cosine_similarity(current_TAC_pred_batch.to(self.device), current_TAC_batch.to(self.device), 0)
          cosine_sim += torch.sum(cosine_sim_slice).item() / len(maskk[maskk>0])

          weights = (self.frame_duration_batch.to(self.device) * torch.exp(-0.00631 * time_stamp_batch)) / (current_TAC_batch.to(self.device))
          weights = torch.nan_to_num(weights, posinf=1)
          chi2_slice = torch.sum(torch.multiply(square, weights), axis=0)
          chi2 += torch.sum(chi2_slice).item() / len(maskk[maskk>0])
        else:
          metric_mse += similaritymeasures_torch.mse(current_TAC_pred_batch.to(self.device), current_TAC_batch.to(self.device)).item()
          metric_mae += similaritymeasures_torch.mae(current_TAC_pred_batch.to(self.device), current_TAC_batch.to(self.device)).item()
          cosine_sim += torch.mean(torch.nn.functional.cosine_similarity(current_TAC_pred_batch.to(self.device), current_TAC_batch.to(self.device), 0)).item()

          square = torch.square(current_TAC_pred_batch.to(self.device) - current_TAC_batch.to(self.device))
          weights = (self.frame_duration_batch.to(self.device) * torch.exp(-0.00631 * time_stamp_batch)) / (current_TAC_batch.to(self.device))
          weights = torch.nan_to_num(weights, posinf=1)
          chi2_slice = torch.sum(torch.multiply(square, weights), axis=0)
          chi2 += torch.mean(chi2_slice).item()
        
        # The following metrics are the same independently from self.config["mask_loss"]
        r2 += similaritymeasures_torch.r2(current_TAC_pred_batch.to(self.device), current_TAC_batch.to(self.device)).item()

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
    #kinetic_params = [4, 192*192, 600]

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
    kinetic_params = kinetic_params.permute((1, 0, 2)) # [192*192, 4, 600]
    k1 = kinetic_params[:, 0, :].unsqueeze(1) # [192*192, 1, 600]
    k2 = kinetic_params[:, 1, :].unsqueeze(1) # [192*192, 1, 600]
    k3 = kinetic_params[:, 2, :].unsqueeze(1) # [192*192, 1, 600]
    Vb = kinetic_params[:, 3, :].unsqueeze(1) # [192*192, 1, 600]

    # Compute the TAC
    current_pred_curve = PET_2TC_KM_batch(idif_batch.to(self.device), self.t_batch.to(self.device), k1.to(self.device), k2.to(self.device), k3.to(self.device), Vb.to(self.device))

    return current_pred_curve, None

  def log_input_target(self, batch, TAC_mes_batch, truth_seg=None, mode="train"):
    if (self.config["experiment"] == "segmentation") or (self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation"):

      fig, axes = plt.subplots(2, len(self.config["segmentation_list"]) + 1, figsize=(20, 20))

      # Plot for TAC_mes_batch images
      axes[0, 0].imshow(TAC_mes_batch[0, 0, 30, :, :].cpu().detach().numpy(), cmap="gray")
      axes[0, 0].axis("off")
      axes[0, 0].set_title("30t")
      axes[0, 1].imshow(TAC_mes_batch[0, 0, 50, :, :].cpu().detach().numpy(), cmap="gray")
      axes[0, 1].axis("off")
      axes[0, 1].set_title("50t")

      # Hide the unused subplots in the first row
      for j in range(2, len(self.config["segmentation_list"]) + 1):
        axes[0, j].axis("off")

      # First subplot for truth_seg
      im1 = axes[1, 0].imshow(truth_seg[0, 0, :, :].cpu().detach().numpy(), cmap="gray")
      axes[1, 0].axis("off")
      axes[1, 0].set_title("Background")
      im1.set_clim(0, 1)
      
      # Rest of the subplots for truth_seg
      for i in range(1, len(self.config["segmentation_list"]) + 1):
        im = axes[1, i].imshow(truth_seg[0, i, :, :].cpu().detach().numpy(), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(self.config["segmentation_list"][i - 1])
        im.set_clim(0, 1)

      fig.suptitle("Patient: " + str(batch[0][0]) + " Slice: " + str(batch[1][0].cpu().detach().numpy()))
    
      plt.tight_layout()
      if mode == "train":
        wandb.log({"Input (training batch)": wandb.Image(fig)})
      elif mode == "validation":
        wandb.log({"Input (validation batch)": wandb.Image(fig)})
      elif mode == "test":
        wandb.log({"Input (test batch)": wandb.Image(fig)})
      else:
        print("ERROR: mode not recognized")
      plt.close()

    
    elif self.config["experiment"] == "kinetics":
      fig, axes = plt.subplots(1, 3, figsize=(15, 5))

      # Plot for TAC_mes_batch images
      axes[0].imshow(TAC_mes_batch[0, 0, 10, :, :].cpu().detach().numpy(), cmap="gray")
      axes[0].axis("off")
      axes[0].set_title("10t")
      axes[1].imshow(TAC_mes_batch[0, 0, 30, :, :].cpu().detach().numpy(), cmap="gray")
      axes[1].axis("off")
      axes[1].set_title("30t")
      axes[2].imshow(TAC_mes_batch[0, 0, 50, :, :].cpu().detach().numpy(), cmap="gray")
      axes[2].axis("off")
      axes[2].set_title("50t")
    
      fig.suptitle("Patient: " + str(batch[0][0]) + " Slice: " + str(batch[1][0].cpu().detach().numpy()))
      
      plt.tight_layout()
      if mode == "train":
        wandb.log({"Input (training batch)": wandb.Image(fig)})
      elif mode == "validation":
        wandb.log({"Input (validation batch)": wandb.Image(fig)})
      elif mode == "test":
        wandb.log({"Input (test batch)": wandb.Image(fig)})
      else:
        print("ERROR: mode not recognized")
      plt.close()

    
  def log_output_kinetics(self, batch, TAC_mes_batch, TAC_pred_batch, kin_logits, mode="train"):
    # ------------------ Log Slice ------------------
    kinetic_params = apply_final_activation(kin_logits, self.config)
    fig = log_slice(self.config, TAC_mes_batch, kinetic_params)
    if mode == "train":
      wandb.log({"Slice (training batch)": wandb.Image(fig)})
    elif mode == "validation":
      wandb.log({"Slice (validation batch)": wandb.Image(fig)})
    elif mode == "test":
      wandb.log({"Slice (test batch)": wandb.Image(fig)})
    plt.close()

    # ------------------ Log TAC ------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))

    # Plot for TAC_mes_batch images
    axes[0, 0].imshow(TAC_mes_batch[0, 0, 10, :, :].cpu().detach().numpy(), cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title("10t")
    axes[0, 1].imshow(TAC_mes_batch[0, 0, 30, :, :].cpu().detach().numpy(), cmap="gray")
    axes[0, 1].axis("off")
    axes[0, 1].set_title("30t")
    axes[0, 2].imshow(TAC_mes_batch[0, 0, 50, :, :].cpu().detach().numpy(), cmap="gray")
    axes[0, 2].axis("off")
    axes[0, 2].set_title("50t")

    # Plot for TAC_pred_batch images
    axes[1, 0].imshow(TAC_pred_batch[0, 0, 10, :, :].cpu().detach().numpy(), cmap="gray")
    axes[1, 0].axis("off")
    axes[1, 0].set_title("10t*")
    axes[1, 1].imshow(TAC_pred_batch[0, 0, 30, :, :].cpu().detach().numpy(), cmap="gray")
    axes[1, 1].axis("off")
    axes[1, 1].set_title("30t*")
    axes[1, 2].imshow(TAC_pred_batch[0, 0, 50, :, :].cpu().detach().numpy(), cmap="gray")
    axes[1, 2].axis("off")
    axes[1, 2].set_title("50t*")
    
    fig.suptitle("Patient: " + str(batch[0][0]) + " Slice: " + str(batch[1][0].cpu().detach().numpy()))
    
    plt.tight_layout()
    if mode == "train":
      wandb.log({"Output TACs (training batch)": wandb.Image(fig)})
    elif mode == "validation":
      wandb.log({"Output TACs (validation batch)": wandb.Image(fig)})
    elif mode == "test":
      wandb.log({"Output TACs (test batch)": wandb.Image(fig)})
    else:
      print("ERROR: mode not recognized")
    plt.close()


  def log_output_segmentation(self, batch, truth_seg, seg_logits, mode="train"):
    fig, axes = plt.subplots(2, len(self.config["segmentation_list"]) + 1, figsize=((len(self.config["segmentation_list"]) + 1) * 2, len(self.config["segmentation_list"]) + 1))
    fig.suptitle("Patient: "+str(batch[0][0])+" Slice: "+str(batch[1][0]))
    for i in range(0, (len(self.config["segmentation_list"]) + 1), 1):
      axes[0, i].imshow(((seg_logits[0, i, :, :] > 0.5) * 255).cpu().detach().numpy(), cmap="binary")
      axes[0, i].axis("off")
      axes[0, i].set_title(self.config["segmentation_list"][i - 1] if i > 0 else "Background")
      axes[1, i].imshow(truth_seg[0, i, :, :].cpu().detach().numpy(), cmap="binary")
      axes[1, i].axis("off")
      axes[1, i].set_title("Truth")
    plt.tight_layout()
    if mode == "train": 
      wandb.log({"Segmentation output (training batch)": wandb.Image(fig)})
    elif mode == "validation":
      wandb.log({"Segmentation output (validation batch)": wandb.Image(fig)})
    elif mode == "test":
      wandb.log({"Segmentation output (test batch)": wandb.Image(fig)})
    plt.close()


  def training_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC, seg, CT]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1)  # adding channel dimension --> [b, 1, 62, w, h]
    x = F.pad(TAC_mes_batch, (0,0,0,0,1,1))   # padding --> [b, 1, 64, w, h]

    if ((self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation2") or (self.config["experiment"] == "segmentation")):
      truth_seg = F.one_hot(batch[3].long(), num_classes= len(self.config["segmentation_list"]) + 1).permute(0, 3, 1, 2).float()
      if self.config["inverse_segmentations"]:
        truth_seg = 1 - truth_seg
    else:
      truth_seg = None

    input = {"batch": batch, "TAC_mes_batch": TAC_mes_batch, "TAC_slice": x}

    #------------------ Log the input and target ------------------#
    if self.current_epoch == 0 and (batch_idx == 0 or batch_idx == 1):
      self.log_input_target(batch, TAC_mes_batch, truth_seg, mode="train")


    #------------------ Forward pass ------------------
    prediction = self.forward_implementation(input)        # [b, 4, 1, w, h]

    #------------------ Process Output Adaptation3 ------------------
    if ((self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation2")):
      #----------------- Log kinetic loss ------------------
      loss_kin = prediction["loss_kin"]
      self.log('train_loss_kinetics', loss_kin.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
      #----------------- Log kinetic metrics ------------------
      metrics_kin = prediction["metrics_kin"]
      self.log('train_mse_kinetics', metrics_kin["mse"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_mae_kinetics', metrics_kin["mae"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_cosine_sim_kinetics', metrics_kin["cosine_sim"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_r2_kinetics', metrics_kin["r2"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      #------------------ Loss segmentation ------------------
      loss_seg = self.seg_loss_function(prediction["seg_logits"], truth_seg)
      dice_score = 1 - self.dice_loss(prediction["seg_logits"], truth_seg)
      #----------------- Log segmentation loss ------------------
      self.log('train_loss_segmentation', loss_seg, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_dice_score', dice_score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      #------------------ Loss ------------------
      combined_loss = self.loss_combination(loss_kin, loss_seg)
      #----------------- Log total loss ------------------
      self.log('train_loss', combined_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      #------------------ Backward pass ------------------
      if self.config["alternate_training"]:
        if batch_idx % 2 == 0:
          if self.config["loss_combination"] == "weighted_sum":
            loss_kin = loss_kin / self.config["rescaling_factor_kinetic_loss"]
          loss_kin.backward(retain_graph=True)
          loss = loss_kin
        else:
          loss_seg.backward(retain_graph=True)
          loss = loss_seg
      else:
        combined_loss.backward(retain_graph=True)
        loss = combined_loss



    #------------------ Process output Segmentation ------------------
    elif self.config["experiment"] == "segmentation":
      #------------------ Loss segmentation ------------------
      loss_seg = self.seg_loss_function(prediction["seg_logits"], truth_seg)
      dice_score = 1 - self.dice_loss(prediction["seg_logits"], truth_seg)
      #----------------- Log segmentation loss ------------------
      self.log('train_loss_segmentation', loss_seg, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_dice_score', dice_score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      loss = loss_seg
      #----------------- Backward pass ------------------
      loss.backward(retain_graph=True)


    #------------------ Process output Kinetics ------------------
    elif self.config["experiment"] == "kinetics":
      #----------------- Log kinetic loss ------------------
      loss_kin = prediction["loss_kin"]
      self.log('train_loss_kinetics', loss_kin, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
      #----------------- Log kinetic metrics ------------------
      metrics_kin = prediction["metrics_kin"]
      self.log('train_mse_kinetics', metrics_kin["mse"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_mae_kinetics', metrics_kin["mae"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_cosine_sim_kinetics', metrics_kin["cosine_sim"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_r2_kinetics', metrics_kin["r2"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      loss = loss_kin
      #------------------ Backward pass ------------------
      loss.backward(retain_graph=True)

    #------------------ Log Gradients ------------------
    if "max_grad_kin" in prediction.keys():
      self.log('max_grad_kin', prediction["max_grad_kin"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
    if "max_grad_seg" in prediction.keys():
      self.log('max_grad_seg', prediction["max_grad_seg"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #------------------ Empty gradients ------------------
    self.zero_grad()

    #------------------ Log the figures ------------------
    if ((np.random.rand() < 0.5) and (self.num_train_log_images < self.config["log_train_imgs"])):
      if ((self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation2")):
        self.log_output_kinetics(batch, TAC_mes_batch, prediction["TAC_pred"], prediction["kin_logits"], mode="train")
        self.log_output_segmentation(batch, truth_seg, prediction["seg_logits"], mode="train")
      if self.config["experiment"] == "segmentation":
        self.log_output_segmentation(batch, truth_seg, prediction["seg_logits"], mode="train")
      if self.config["experiment"] == "kinetics":
        self.log_output_kinetics(batch, TAC_mes_batch, prediction["TAC_pred"], prediction["kin_logits"], mode="train")
      
      self.num_train_log_images += 1

    return {"loss": loss}


  def on_train_epoch_end(self):
    self.num_train_log_images = 0
    self.max_grad_kin = 0
    self.max_grad_seg = 0
    return 

  def validation_step(self, batch, batch_idx):
    with torch.no_grad():
      # batch = [patient, slice, TAC, seg, CT]
      TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
      x = torch.nn.functional.pad(TAC_mes_batch, (0,0,0,0,1,1), "replicate")   # padding --> [b, 1, 64, w, h]

      if ((self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation2") or (self.config["experiment"] == "segmentation")):
        truth_seg = F.one_hot(batch[3].long(), num_classes= len(self.config["segmentation_list"]) + 1).permute(0, 3, 1, 2).float()
        if self.config["inverse_segmentations"]:
          truth_seg = 1 - truth_seg
      else:
        truth_seg = None

      input = {"batch": batch, "TAC_mes_batch": TAC_mes_batch, "TAC_slice": x}

      #------------------ Log the input and target ------------------
      if self.current_epoch == 0 and (batch_idx == 0 or batch_idx == 1):
        self.log_input_target(batch, TAC_mes_batch, truth_seg, mode="validation")

      #------------------ Forward pass ------------------
      prediction = self.forward_implementation(input)        # [b, 4, 1, w, h]

      #------------------ Process Combination of outputs ------------------
      if ((self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation2")):
        #----------------- Log kinetic loss ------------------
        loss_kin = prediction["loss_kin"]
        self.log('val_loss_kinetics', loss_kin, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #----------------- Log kinetic metrics ------------------
        metrics_kin = prediction["metrics_kin"]
        self.log('val_mse_kinetics', metrics_kin["mse"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_mae_kinetics', metrics_kin["mae"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_cosine_sim_kinetics', metrics_kin["cosine_sim"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_r2_kinetics', metrics_kin["r2"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #------------------ Loss segmentation ------------------
        loss_seg = self.seg_loss_function(prediction["seg_logits"], truth_seg)
        dice_score = 1 - self.dice_loss(prediction["seg_logits"], truth_seg)
        #----------------- Log segmentation loss ------------------
        self.log('val_loss_segmentation', loss_seg, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_dice_score', dice_score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #------------------ Loss ------------------
        loss = self.loss_combination(loss_kin, loss_seg)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

      
      elif self.config["experiment"] == "segmentation":
        #------------------ Loss segmentation ------------------
        loss_seg = self.seg_loss_function(prediction["seg_logits"], truth_seg)
        dice_score = 1 - self.dice_loss(prediction["seg_logits"], truth_seg)
        #----------------- Log segmentation loss ------------------
        self.log('val_loss_segmentation', loss_seg, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_dice_score', dice_score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss = loss_seg

      
      elif self.config["experiment"] == "kinetics":
        #----------------- Log kinetic loss ------------------
        loss_kin = prediction["loss_kin"]
        self.log('val_loss_kinetics', loss_kin, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #----------------- Log kinetic metrics ------------------
        metrics_kin = prediction["metrics_kin"]
        self.log('val_mse_kinetics', metrics_kin["mse"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_mae_kinetics', metrics_kin["mae"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_cosine_sim_kinetics', metrics_kin["cosine_sim"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_r2_kinetics', metrics_kin["r2"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss = loss_kin


      
      #------------------ Log the figures ------------------
      if ((np.random.rand() < 0.5) and (self.num_val_log_images < self.config["log_val_imgs"])):
        if ((self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation2")):
          self.log_output_kinetics(batch, TAC_mes_batch, prediction["TAC_pred"], prediction["kin_logits"], mode="validation")
          self.log_output_segmentation(batch, truth_seg, prediction["seg_logits"], mode="validation")
        if self.config["experiment"] == "segmentation":
          self.log_output_segmentation(batch, truth_seg, prediction["seg_logits"], mode="validation")
        if self.config["experiment"] == "kinetics":
          self.log_output_kinetics(batch, TAC_mes_batch, prediction["TAC_pred"], prediction["kin_logits"], mode="validation")
      
        self.num_val_log_images += 1
      
      return {'val_loss': loss}
  

  def on_validation_epoch_end(self):
    self.num_val_log_images = 0
    return

  
  def test_step(self, batch, batch_idx):
    with torch.no_grad():
      # batch = [patient, slice, TAC, seg, CT]
      patients_in_batch = batch[0]
      slices_in_batch = batch[1]
      TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
      x = torch.nn.functional.pad(TAC_mes_batch, (0,0,0,0,1,1), "replicate")   # padding --> [b, 1, 64, w, h]

      if ((self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation2") or (self.config["experiment"] == "segmentation")):
        truth_seg = F.one_hot(batch[3].long(), num_classes= len(self.config["segmentation_list"]) + 1).permute(0, 3, 1, 2).float()
        if self.config["inverse_segmentations"]:
          truth_seg = 1 - truth_seg
      else:
        truth_seg = None

      input = {"batch": batch, "TAC_mes_batch": TAC_mes_batch, "TAC_slice": x}

      #------------------ Prepare the paths ------------------
      current_run_name = wandb.run.name
      resume_run_name = os.path.split(os.path.split(self.trainer.ckpt_path)[0])[1]
      self.img_path, self.pd_path, self.pt_path, self.nifty_path = make_save_folder_struct(current_run_name, resume_run_name, self.config["full_resume_path"], self.trainer.ckpt_path)

      #------------------ Log the input and target ------------------
      if self.current_epoch == 0 and (batch_idx == 0 or batch_idx == 1):
        self.log_input_target(batch, TAC_mes_batch, truth_seg, mode="test")

      #------------------ Forward pass ------------------
      prediction = self.forward_implementation(input)

      if ((self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation2")):
        #----------------- Log segmentation dice scores ------------------
        num_classes = prediction["seg_logits"].shape[1]
        dice_scores = []
        for i in range(num_classes):
            dice_score = 1 - self.dice_loss(prediction["seg_logits"][:, i:i+1], truth_seg[:, i:i+1])
            dice_scores.append(dice_score)
            self.log(f'test_dice_score_class_{i}', dice_score, on_step=True, on_epoch=False)
        
        # Calculate and log overall dice score
        overall_dice_score = 1 - self.dice_loss(prediction["seg_logits"], truth_seg)
        self.log('test_overall_dice_score', overall_dice_score, on_step=True, on_epoch=False)
        #------------------ Save the kin results locally ------------------
        to_save = [patients_in_batch, slices_in_batch, prediction["kin_logits"]]
        torch.save(to_save, os.path.join(self.pt_path, "kin_P_"+str(patients_in_batch[0])+"_B_"+str(batch_idx)+".pt"))

        #------------------ Save the seg results locally ------------------
        to_save = [patients_in_batch, slices_in_batch, prediction["seg_logits"]]
        torch.save(to_save, os.path.join(self.pt_path, "seg_P_"+str(patients_in_batch[0])+"_B_"+str(batch_idx)+".pt"))

        #------------------ Save Metrics locally ------------------
        for i in range(len(patients_in_batch)):
          p = patients_in_batch[i]
          if not p in self.summary.keys(): 
            self.summary[p] = dict()
            self.patient_totals[p] = {
                "MSE": 0, "MAE": 0, "Cosine": 0, "R2": 0, 
                "Dice": 0
            }
            for i in range(len(self.config["segmentation_list"]) + 1):
                self.patient_totals[p][f"Dice_class_{i}"] = 0
            self.patient_counts[p] = 0
          for j in range(len(slices_in_batch)):
            s = int(slices_in_batch[j].item())
            if patients_in_batch[j] == p:
              self.summary[p][s] = dict()
              self.summary[p][s]["MSE"] = prediction["metrics_kin"]["mse"]
              self.summary[p][s]["MAE"] = prediction["metrics_kin"]["mae"]
              self.summary[p][s]["Cosine"] = prediction["metrics_kin"]["cosine_sim"]
              self.summary[p][s]["R2"] = prediction["metrics_kin"]["r2"]
              self.summary[p][s]["Dice"] = overall_dice_score.item()
              for i in range(num_classes):
                self.summary[p][s][f"Dice_class_{i}"] = dice_scores[i].item()

              self.patient_counts[p] += 1
              self.patient_totals[p]["MSE"] += prediction["metrics_kin"]["mse"]
              self.patient_totals[p]["MAE"] += prediction["metrics_kin"]["mae"]
              self.patient_totals[p]["Cosine"] += prediction["metrics_kin"]["cosine_sim"]
              self.patient_totals[p]["R2"] += prediction["metrics_kin"]["r2"]
              self.patient_totals[p]["Dice"] += overall_dice_score.item()
              for i in range(num_classes):
                self.patient_totals[p][f"Dice_class_{i}"] += dice_scores[i].item()

      #------------------ Process Kinetic Output ------------------
      if self.config["experiment"] == "kinetics":
        #------------------ Save the results locally ------------------
        to_save = [patients_in_batch, slices_in_batch, prediction["kin_logits"]]
        torch.save(to_save, os.path.join(self.pt_path, "P_"+str(patients_in_batch[0])+"_B_"+str(batch_idx)+".pt"))

        #------------------ Save Metrics locally ------------------
        for i in range(len(patients_in_batch)):
          p = patients_in_batch[i]
          if not p in self.summary.keys(): 
            self.summary[p] = dict()
            self.patient_totals[p] = {"MSE": 0, "MAE": 0, "Cosine": 0, "R2": 0}
            self.patient_counts[p] = 0
          for j in range(len(slices_in_batch)):
            s = int(slices_in_batch[j].item())
            if patients_in_batch[j] == p:
              self.summary[p][s] = dict()
              self.summary[p][s]["MSE"] = prediction["metrics_kin"]["mse"]
              self.summary[p][s]["MAE"] = prediction["metrics_kin"]["mae"]
              self.summary[p][s]["Cosine"] = prediction["metrics_kin"]["cosine_sim"]
              self.summary[p][s]["R2"] = prediction["metrics_kin"]["r2"]
              self.patient_counts[p] += 1
              self.patient_totals[p]["MSE"] += prediction["metrics_kin"]["mse"]
              self.patient_totals[p]["MAE"] += prediction["metrics_kin"]["mae"]
              self.patient_totals[p]["Cosine"] += prediction["metrics_kin"]["cosine_sim"]
              self.patient_totals[p]["R2"] += prediction["metrics_kin"]["r2"]

      #------------------ Process Segmentation Output ------------------
      elif self.config["experiment"] == "segmentation":
        #----------------- Log segmentation dice scores ------------------
        dice_scores = 1 - self.dice_loss(prediction["seg_logits"], truth_seg, return_per_class=True)
        overall_dice_score = dice_scores.mean()
        self.log('test_dice_score', overall_dice_score, on_step=True, on_epoch=False)
        num_classes = dice_scores.shape[0]
        for i in range(num_classes):
          self.log(f'test_dice_score_class_{i}', dice_scores[i], on_step=True, on_epoch=False)
        
        #------------------ Save the results locally ------------------
        to_save = [patients_in_batch, slices_in_batch, prediction["seg_logits"]]
        torch.save(to_save, os.path.join(self.pt_path, "P_"+str(patients_in_batch[0])+"_B_"+str(batch_idx)+".pt"))

        #------------------ Save Metrics locally ------------------
        for i in range(len(patients_in_batch)):
          p = patients_in_batch[i]
          if not p in self.summary.keys(): 
            self.summary[p] = dict()
            self.patient_totals[p] = {"Dice": 0}
            for i in range(num_classes):
                self.patient_totals[p][f"Dice_class_{i}"] = 0
            self.patient_counts[p] = 0
          for j in range(len(slices_in_batch)):
            s = int(slices_in_batch[j].item())
            if patients_in_batch[j] == p:
              self.summary[p][s] = dict()
              self.summary[p][s]["Dice"] = overall_dice_score.item()
              for i in range(num_classes):
                self.summary[p][s][f"Dice_class_{i}"] = dice_scores[i].item()
              self.patient_counts[p] += 1
              self.patient_totals[p]["Dice"] += overall_dice_score.item()
              for i in range(num_classes):
                self.patient_totals[p][f"Dice_class_{i}"] += dice_scores[i].item()

      #------------------ Log the figures ------------------
      if batch_idx % 50 == 0:
        if ((self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation2")):
          self.log_output_kinetics(batch, TAC_mes_batch, prediction["TAC_pred"], prediction["kin_logits"], mode="test")
          self.log_output_segmentation(batch, truth_seg, prediction["seg_logits"], mode="test")
        if self.config["experiment"] == "segmentation":
          self.log_output_segmentation(batch, truth_seg, prediction["seg_logits"], mode="test")
        if self.config["experiment"] == "kinetics":
          self.log_output_kinetics(batch, TAC_mes_batch, prediction["TAC_pred"], prediction["kin_logits"], mode="test")

      return


  
  def on_test_epoch_end(self):
    run_name = os.path.split(os.path.split(self.pd_path)[0])[1]
    patient_std_devs = {}
    for p in self.summary.keys():
      current_df = pd.DataFrame.from_dict(self.summary[p])
      # This file contains the metrics per slice. It allows to identify slices with bad peformance. 
      # It is also used during evaluation phase to compute the metrics on the whole dataset
      current_df.to_excel(os.path.join(self.pd_path, p + "_metric_per_slice_" + run_name + ".xlsx"))
      
      patient_metrics = pd.DataFrame.from_dict(self.summary[p]).transpose()
      patient_means = patient_metrics.mean()
      squared_diffs = (patient_metrics - patient_means) ** 2
      patient_std_devs[p] = np.sqrt(squared_diffs.mean()).to_dict()

    # Compute the average metrics per patient
    patient_averages = {p: {metric: total / self.patient_counts[p] for metric, total in totals.items()} for p, totals in self.patient_totals.items()}
    # Compute the average metrics over the whole dataset
    if self.config["experiment"] == "segmentation":
        metric_list = ["Dice"] + [f"Dice_class_{i}" for i in range(num_classes)]
        overall_averages = {metric: sum(patient_averages[p][metric] for p in patient_averages) / len(patient_averages) for metric in metric_list}
        overall_sums_of_squares = {metric: 0 for metric in metric_list}
    elif self.config["experiment"] == "kinetics":
        metric_list = ["MSE", "MAE", "Cosine", "R2"]
        overall_averages = {metric: sum(patient_averages[p][metric] for p in patient_averages) / len(patient_averages) for metric in metric_list}
        overall_sums_of_squares = {metric: 0 for metric in metric_list}
    elif ((self.config["experiment"] == "adaptation3") or (self.config["experiment"] == "adaptation2")):
        metric_list = ["MSE", "MAE", "Cosine", "R2", "Dice"] + [f"Dice_class_{i}" for i in range(num_classes)]
        overall_averages = {metric: sum(patient_averages[p][metric] for p in patient_averages) / len(patient_averages) for metric in metric_list}
        overall_sums_of_squares = {metric: 0 for metric in metric_list}

    for p in patient_averages.keys():
      for metric in metric_list:
        overall_sums_of_squares[metric] += (patient_averages[p][metric] - overall_averages[metric]) ** 2

    overall_std_devs = {metric: np.sqrt(total / len(patient_averages)) for metric, total in overall_sums_of_squares.items()}

    print(f"Patient averages: {patient_averages}")
    print(f"Patient standard deviations: {patient_std_devs}")
    print(f"Overall standard deviations: {overall_std_devs}")
    print(f"Overall averages: {overall_averages}")

    print(f"Saving metrics to {self.pd_path}...")

    #Save metrics to a file
    with open(os.path.join(self.pd_path, "patient_averages_" + run_name + ".json"), "w") as f:
      json.dump(patient_averages, f)

    with open(os.path.join(self.pd_path, "patient_std_devs_" + run_name + ".json"), "w") as f:
      json.dump(patient_std_devs, f)
    
    with open(os.path.join(self.pd_path, "overall_averages_" + run_name + ".json"), "w") as f:
      json.dump(overall_averages, f)
    
    with open(os.path.join(self.pd_path, "overall_std_devs_" + run_name + ".json"), "w") as f:
      json.dump(overall_std_devs, f)

    return 

  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    optimizer.step(closure=optimizer_closure)
    self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
    if (epoch > self.config["val_freq"] and batch_idx == 0):
      if self.config["lr_scheduler"] == "ReduceLROnPlateau":
        self.scheduler.step(self.trainer.callback_metrics[self.config["monitor_metric"]])
      else:
        self.scheduler.step()


  def configure_optimizers(self):
    # Define the optimizer
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"], weight_decay=self.config['weight_decay'])#, betas=(self.config['beta1'], self.config['beta2']))
    
    # Define the learning rate scheduler
    if self.config["lr_scheduler"] == "CosineAnnealingLR":
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['cosine_T_max'], eta_min=self.config['cosine_eta_min'], last_epoch=-1)

    elif self.config["lr_scheduler"] == "MultiStepLR":
      self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['multistep_milestones'], gamma=self.config['multistep_gamma'], last_epoch=-1)

    elif self.config["lr_scheduler"] == "ReduceLROnPlateau":
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.config['reduce_factor'], patience=self.config['reduce_patience'], verbose=True)

    return [optimizer]