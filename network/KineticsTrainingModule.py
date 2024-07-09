import torch

from network.unet_blocks import UNet
from network.unet_blocks_ST import UNet_ST
from network.unet_blocks_ST_dropout import UNet_ST_dropout

from utils.utils_logging import mask_data

from .BaseTrainingModule import BaseTrainingModule

class KineticsTrainingModule(BaseTrainingModule):
  def __init__(self, config):
    super(KineticsTrainingModule, self).__init__(config)

    #------------ Kinetics Network ------------
    if self.config["use_spatio_temporal_unet"]:
      #self.model = UNet_ST(in_channels=1, out_channels=self.config["output_size"], config=self.config)
      self.model = UNet_ST_dropout(in_channels=1, out_channels=self.config["output_size"], config=self.config)
    else:
      self.model = UNet(in_channels=1, out_channels=self.config["output_size"], config=self.config)

    self.model.train()
    self.model.to(self.device)

    #------------ Initialize Weights ------------
    if self.config["init_weights"]:
      self.initialize_weights(self.model)

    #------------ Other Parameters ------------
    self.max_grad_kin = 0

  def forward_implementation(self, input):
    kin_logits = self.model(input["TAC_slice"])

    #----------------- Process Kinetic Output ------------------
    loss_dict, metric_dict, TAC_pred_batch = self.accumulate_loss_and_metric(batch=input["batch"], logits=kin_logits)

    if self.config["mask_loss"]:
      input["TAC_mes_batch"], TAC_pred_batch, kin_out = mask_data(input["TAC_mes_batch"], TAC_pred_batch, kin_logits, self.time_stamp, patch_size=self.patch_size)
   
    #------------------ Calculate Gradient Kinetics ------------------
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        if param.grad is not None:
          self.max_grad_kin = max(self.max_grad_kin, param.grad.abs().max().item())

    return {"kin_out": kin_out, "kin_logits": kin_logits, "TAC_pred": TAC_pred_batch, "loss": loss_dict["loss"], "max_grad_kin": self.max_grad_kin, "metrics": metric_dict}


  def load_from_checkpoint(self, kinetic_ckpt=None, full_ckpt=None):
    if (kinetic_ckpt is not None):
      print("Error: Kinetic checkpoint was provided but full checkpoint should be given for kinetic experiment.")
      exit(1)

    if full_ckpt is None:
      print("Training from scratch.")
      return

    print("Resuming training from: ", full_ckpt)
    weights = torch.load(full_ckpt)
    for name, param in self.model.named_parameters():
      if ("model." + name) in weights["state_dict"].keys():
        param.data = weights["state_dict"]["model." + name].data
      else:
        print("\tParameter not found: ", name)
