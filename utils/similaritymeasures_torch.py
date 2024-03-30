import torch
from torcheval.metrics.functional import r2_score

def mae(data1, data2):            # Mean Absolute Error
    c = torch.abs(data1 - data2)
    return torch.mean(c)

def mse(data1, data2):            # Mean Squared Error
    c = torch.square(data1 - data2)
    return torch.mean(c)

def r2(prediction, ground_truth):
  score = r2_score(prediction, ground_truth)
  return torch.nan_to_num(score, nan=1, neginf=0)

def dice(prediction, ground_truth, epsilon=1e-6):
  intersection = torch.sum(prediction * ground_truth)
  union = torch.sum(prediction) + torch.sum(ground_truth)
  return (2 * intersection + epsilon) / (union + epsilon)