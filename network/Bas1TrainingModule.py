from monai.networks.nets import UNet as UNet_monai
from monai.networks.nets import SegResNetDS as SegResNetDS_monai

from .BaseTrainingModule import BaseTrainingModule

class Bas1TrainingModule(BaseTrainingModule):
  def __init__(self, config):
    super(Bas1TrainingModule, self).__init__(config)

    #------------ Segmentation Network ------------
    if self.config["segmentation_network"] == "UNet":
      self.model = UNet_monai(spatial_dims = 2, in_channels=64, out_channels=self.config["output_size_seg"], 
                                      channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3, 
                                      num_res_units=2, act='PRELU', norm='INSTANCE', 
                                      dropout=self.config["dropout"], bias=True, adn_ordering='NDA')
    
    elif self.config["segmentation_network"] == "SegResNetDS":
      self.model = SegResNetDS_monai(spatial_dims=2, init_filters=32, 
                                      in_channels=64, out_channels=self.config["output_size_seg"], 
                                      act='relu', norm='batch', 
                                      blocks_down=(1, 2, 2, 4), blocks_up=None, dsdepth=1, 
                                      preprocess=None, upsample_mode='deconv', resolution=None)
                                    
    #self.model = UNet_2D(in_channels=self.config["output_size"], out_channels=self.config["output_size_seg"], config=self.config)

    self.model.train()
    self.model.to(self.device)

    #------------ Initialize Weights ------------
    if self.config["init_weights"]:
      self.initialize_weights(self.model)

    #------------ Other Parameters ------------
    self.max_grad_seg = 0

  def forward_implementation(self, input):
    input["TAC_slice"] = input["TAC_slice"].squeeze(1)
    seg_logits = self.model(input["TAC_slice"])
    seg_output = seg_logits.argmax(dim=1)

    #------------------ Calculate Gradient Kinetics ------------------
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        if param.grad is not None:
          self.max_grad_seg = max(self.max_grad_seg, param.grad.abs().max().item())

      

    return {"seg_logits": seg_logits, "max_grad_seg": self.max_grad_seg, "seg_output": seg_output}

  def load_from_checkpoint(self, kinetic_ckpt=None, full_ckpt=None):
    if kinetic_ckpt is not None:
      print("Error: Kinetic checkpoint was provided but full checkpoint should be given for Baseline 1 experiment.")
      exit(1)

    # Load all weights
    if full_ckpt is not None:
      print("Resuming training from: ", full_ckpt)
      weights = torch.load(full_ckpt)
      for name, param in self.model.named_parameters():
        if name in weights["state_dict"].keys():
          param.data = weights["state_dict"][name].data
        else:
          print("\tParameter not found: ", name)