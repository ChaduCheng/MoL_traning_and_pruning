

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models


device = 'cuda' if torch.cuda.is_available() else 'cpu'



loader = transforms.Compose([
  transforms.ToTensor()]) 
 
unloader = transforms.ToPILImage()


## PIL to tensor


def PIL_to_tensor(image):
 # image = loader(image).unsqueeze(0)
  image = loader(image)
  return image.to(device, torch.float)


def tensor_to_PIL(tensor):
  image = tensor.cpu().clone()
  #image = image.squeeze(0)
  image = unloader(image)
  return image