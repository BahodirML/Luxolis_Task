import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

## Fisheye Transformation
def fisheye(height, width, center, magnitude):
    xx, yy = torch.linspace(-1, 1, width), torch.linspace(-1, 1, height)
    gridy, gridx = torch.meshgrid(yy, xx)   
    grid = torch.stack([gridx, gridy], dim=-1)
    d = center - grid         
    d_sum = torch.sqrt((d**2).sum(axis=-1)) 
    grid += d * d_sum.unsqueeze(-1) * magnitude 
    return grid.unsqueeze(0)   

#Transforming
def image_transform(img):
    transform = transforms.Compose([transforms.ToTensor()])
    tfms_img = transform(img)
    imgs = torch.unsqueeze(tfms_img, dim=0)
    return imgs

#plotting the result
def plot_fisheye(img, fisheye_output):
    fisheye_out = fisheye_output[0].numpy()
    fisheye_out = np.moveaxis(fisheye_out, 0, -1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(img)
    ax[1].imshow(fisheye_out)

    ax[0].set_title('Fisheye Image')
    ax[1].set_title('Flattened image')
    plt.show()


    # In order to save the image I converted the tensor to PIL image
    fisheye_img = Image.fromarray((fisheye_out * 255).astype(np.uint8))
    fisheye_img = ImageOps.autocontrast(fisheye_img)  # Adjust contrast
    fisheye_img.save('fisheyee.jpg')


img = Image.open('fisheye.jpg')
imgs = image_transform(img)
N, C, H, W = imgs.shape
fisheye_grid = fisheye(H, W, torch.tensor([0, 0]), 0.4)

fisheye_output = F.grid_sample(imgs, fisheye_grid, align_corners=True)
plot_fisheye(img, fisheye_output)
