import torch

from network.unet_blocks_ST_2Heads import UNet_ST_2Heads

from utils.utils_logging import mask_data

from .BaseTrainingModule import BaseTrainingModule

class Seg3TrainingModule(BaseTrainingModule):
  def __init__(self, config):
    super(Seg3TrainingModule, self).__init__(config)

    #------------ Kinetics Network ------------
    self.model = UNet_ST_2Heads(in_channels=1, out_channels_kin=self.config["output_size"], out_channels_seg=self.config["output_size_seg"], config=self.config)

    self.model.train()
    self.model.to(self.device)

    #------------ Initialize Weights ------------
    if self.config["init_weights"]:
      self.initialize_weights(self.model)

    #------------ Other Parameters ------------
    self.max_grad_seg = 0
    self.max_grad_kin = 0


  def forward_implementation(self, input):
    kin_logits, seg_logits = self.model(input["TAC_slice"])

    #------------------ Segmentation Output ------------------
    seg_out = seg_logits.argmax(dim=1)
    seg_out = seg_out.squeeze(2)
    seg_logits = seg_logits.squeeze(2)

    #------------------ Kinetic Output ------------------
    loss_dict, metric_dict, TAC_pred_batch = self.accumulate_loss_and_metric(batch=input["batch"], logits=kin_logits)
    if self.config["mask_loss"]:
      _, _, kin_out = mask_data(input["TAC_mes_batch"], TAC_pred_batch, kin_logits, self.time_stamp, patch_size=self.patch_size)

    #------------------ Calculate Gradient ------------------
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        if param.grad is not None:
          if "kin" in name:
            self.max_grad_kin = max(self.max_grad_kin, param.grad.abs().max().item())
          elif "seg" in name:
            self.max_grad_seg = max(self.max_grad_seg, param.grad.abs().max().item())
    
    return {"kin_logits": kin_logits, "kin_out": kin_out, "seg_logits": seg_logits, "seg_out": seg_out, "TAC_pred": TAC_pred_batch, "loss_kin": loss_dict["loss"], "metrics_kin": metric_dict,  "max_grad_kin": self.max_grad_kin, "max_grad_seg": self.max_grad_seg}



  def load_from_checkpoint(self, kinetic_ckpt=None, full_ckpt=None):
    if kinetic_ckpt:
      # Load kinetic weights
      print("Resuming training from kinetic ckpt: ", kinetic_ckpt)
      weights = torch.load(kinetic_ckpt)
      for name, param in self.model.named_parameters():
        if ("model." + name) in weights["state_dict"].keys():
          param.data = weights["state_dict"]["model." + name].data
          if self.config["freeze_kin"]:
            param.requires_grad = False
        elif (("decoder_kin." in name) and ("model." + name.replace("decoder_kin.", "decoder.")) in weights["state_dict"].keys()):
          param.data = weights["state_dict"]["model." + name.replace("decoder_kin.", "decoder.")].data
          if self.config["freeze_kin"]:
            param.requires_grad = False
        else:
          print("\tParameter not found: model." + name)
    elif full_ckpt:
      # Load all weights
      print("Resuming training from: ", full_ckpt)
      weights = torch.load(full_ckpt)
      for name, param in self.model.named_parameters():
        if name in weights["state_dict"].keys():
          param.data = weights["state_dict"][name].data
        else:
          print("\tParameter not found: ", name)
    else:
      print("Training from scratch.")


