from monai.networks.nets import UNet as UNet_monai
from monai.networks.nets import SegResNetDS as SegResNetDS_monai

from network.unet_blocks import UNet
from network.unet_blocks_ST import UNet_ST
from network.unet_blocks_2D import UNet_2D

from .BaseTrainingModule import BaseTrainingModule

class Seg1TrainingModule(BaseTrainingModule):
  def __init__(self, config):
    config["output_size"] = 4 + len(config["segmentation_list"]) + 1   # This is the number of output channels (1 per kinetic parameter + 1 per Organ)
    super(SpaceTempUNetSeg2, self).__init__(config)

    #------------ Kinetics Network ------------
    if self.config["use_spatio_temporal_unet"]:
      self.model = UNet_ST(in_channels=1, out_channels=self.config["output_size"], config=self.config)
    else:
      self.model = UNet(in_channels=1, out_channels=self.config["output_size"], config=self.config)

    #------------ Initialize Weights ------------
    if self.config["init_weights"]:
      self.initialize_weights(self.model)


  def forward_implementation(self, input):
    logits = self.model(input["TAC_slice"])
    kinetics_out, segmentation_out = logits[:, :4], logits[:, 4:]
    return kinetics_out, segmentation_out


  def load_from_checkpoint(self, kinetic_ckpt=None, full_ckpt=None):
    # Load kinetic weights
    if kinetic_ckpt is not None:
      print("Resuming training from kinetic imaging train: ", kinetic_ckpt)
      weights = torch.load(kinetic_ckpt)
      for name, param in self.model.named_parameters():
        if ((name in weights["state_dict"].keys()) and ("final_conv" not in name)):
          param.data = weights["state_dict"][name].data
        elif "final_conv" in name:
          print("\tSkipping final_conv")
        else:
          print("\tParameter not found: ", name)

    # Load all weights
    if full_ckpt is not None:
      print("Resuming training from: ", full_ckpt)
      weights = torch.load(full_ckpt)
      for name, param in self.model.named_parameters():
        if name in weights["state_dict"].keys():
          param.data = weights["state_dict"][name].data
        else:
          print("\tParameter not found: ", name)
