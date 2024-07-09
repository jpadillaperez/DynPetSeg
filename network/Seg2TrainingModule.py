import torch
from monai.networks.nets import UNet as UNet_monai
from monai.networks.nets import SegResNetDS as SegResNetDS_monai

from network.unet_blocks import UNet
from network.unet_blocks_ST import UNet_ST
from network.unet_blocks_ST_dropout import UNet_ST_dropout
from network.unet_blocks_2D import UNet_2D

from utils.utils_logging import mask_data

from .BaseTrainingModule import BaseTrainingModule


class Seg2TrainingModule(BaseTrainingModule):
  def __init__(self, config):
    super(Seg2TrainingModule, self).__init__(config)

    #------------ Kinetics Network ------------
    if self.config["use_spatio_temporal_unet"]:
      self.model_kinetics = UNet_ST_dropout(in_channels=1, out_channels=self.config["output_size"], config=self.config)
    else:
      self.model_kinetics = UNet(in_channels=1, out_channels=self.config["output_size"], config=self.config)
    

    if self.config["freeze_kin"]:
      self.model_kinetics.eval()
      self.model_kinetics.requires_grad = False
    else:
      self.model_kinetics.train()
      self.model_kinetics.requires_grad = True
    
    self.model_kinetics.to(self.device)

    #------------ Segmentation Network ------------
    if self.config["segmentation_network"] == "UNet":
      self.model_segmentation = UNet_monai(spatial_dims = 2, in_channels=self.config["output_size"], out_channels=self.config["output_size_seg"], 
                                      channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3, 
                                      num_res_units=2, act='PRELU', norm='INSTANCE', 
                                      dropout=self.config["dropout_seg"], bias=True, adn_ordering='NDA')
    
    elif self.config["segmentation_network"] == "SegResNetDS":
      self.model_segmentation = SegResNetDS_monai(spatial_dims=2, init_filters=32, 
                                      in_channels=self.config["output_size"], out_channels=self.config["output_size_seg"], 
                                      act='relu', norm='batch', 
                                      blocks_down=(1, 2, 2, 4), blocks_up=None, dsdepth=1, 
                                      preprocess=None, upsample_mode='deconv', resolution=None)

    self.model_segmentation.train()
    self.model_segmentation.to(self.device)
                         
    ##------------ Initialize Weights ------------
    if self.config["init_weights"]:
      #self.initialize_weights(self.model_kinetics)
      self.initialize_weights(self.model_segmentation)

    #------------ Other Parameters ------------
    self.max_grad_seg = 0
    self.max_grad_kin = 0


  def forward_implementation(self, input):
    #----------------- Forward Pass Kinetics ------------------
    kin_logits = self.model_kinetics(input["TAC_slice"])
    #----------------- Process Kinetic Output ------------------
    loss_dict, metric_dict, TAC_pred_batch = self.accumulate_loss_and_metric(batch=input["batch"], logits=kin_logits)
    if self.config["mask_loss"]:
      _, _, kin_out = mask_data(input["TAC_mes_batch"], TAC_pred_batch, kin_logits, self.time_stamp, patch_size=self.patch_size)
    #----------------- Forward Pass segmentation ------------------
    seg_logits = self.model_segmentation(kin_out[:, :, 0, :, :])
    #------------------ Process Segmentation Output ------------------
    seg_out = self.softmax(seg_logits)
    seg_out = torch.argmax(seg_out, dim=1)
    #------------------ Calculate Gradient Kinetics ------------------
    for name, param in self.model_kinetics.named_parameters():
      if param.requires_grad:
        if param.grad is not None:
          self.max_grad_kin = max(max_grad_kin, param.grad.abs().max().item())
    #------------------ Calculate Gradient Segmentation ------------------
    for name, param in self.model_segmentation.named_parameters():
      if param.requires_grad:
        if param.grad is not None:
          self.max_grad_seg = max(max_grad_seg, param.grad.abs().max().item())
  
    return {"kin_logits": kin_logits, "kin_out": kin_out, "seg_logits": seg_logits, "seg_out": seg_out, "TAC_pred": TAC_pred_batch, "loss_kin": loss_dict["loss"].item(), "metrics_kin": metric_dict,  "max_grad_kin": self.max_grad_kin, "max_grad_seg": self.max_grad_seg}



  def load_from_checkpoint(self, kinetic_ckpt=None, full_ckpt=None):
    # Load kinetic weights
    if kinetic_ckpt is not None:
      print("Resuming training from kinetic imaging train: ", kinetic_ckpt)
      weights = torch.load(kinetic_ckpt)
      for name, param in self.model_kinetics.named_parameters():
        if ("model." + name) in weights["state_dict"].keys():
          param.data = weights["state_dict"]["model." + name].data
          param.requires_grad = False
        else:
          print("\tParameter not found: model." + name)

    # Load all weights
    elif full_ckpt is not None:
      print("Resuming full training from: ", full_ckpt)
      weights = torch.load(full_ckpt)
      for name, param in self.model_kinetics.named_parameters():
        if name in weights["state_dict"].keys():
          param.data = weights["state_dict"][name].data
        else:
          print("\tParameter not found: ", name)
      
      for name, param in self.model_segmentation.named_parameters():
        if name in weights["state_dict"].keys():
          param.data = weights["state_dict"][name].data
        else:
          print("\tParameter not found: ", name)

    else:
      print("Training from scratch.")

