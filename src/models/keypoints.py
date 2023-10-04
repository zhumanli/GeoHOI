import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2

restnet_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
recon_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def GetImgLabel(resnet, centroids, seg_masks):
    seg_masks = seg_masks.to('cuda')
    resized_imgs = F.interpolate(seg_masks, size=(224, 224), mode='bilinear', align_corners=False)
    rgb_imgs = torch.cat((resized_imgs, resized_imgs, resized_imgs), dim = 1)
    
    mask_features = resnet(restnet_transform(rgb_imgs))
    
    bs = mask_features.shape[0]
    centroids = centroids.unsqueeze(0).repeat(bs, 1, 1)
    mask_features = mask_features.unsqueeze(1)
    distances = torch.norm(mask_features - centroids, dim=2)
    labels = torch.argmin(distances, dim=1)

    return labels

def GetKeypoints(recon_model, resnet, centroids, seg_masks):
    # seg_masks = seg_masks.to('cuda')
    resized_imgs = F.interpolate(seg_masks, size=(128, 128), mode='bilinear', align_corners=False)
    rgb_imgs = torch.cat((resized_imgs, resized_imgs, resized_imgs), dim = 1)
    
    kernel = torch.ones((3, 3, 3, 3)).to('cuda')
    dilated_imgs = F.conv2d(rgb_imgs, kernel, padding=1)

    labels = GetImgLabel(resnet, centroids, seg_masks)
    recon_input = {'img': recon_transform(dilated_imgs), 'label': labels}

    if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    with torch.no_grad():
        out_recon = recon_model.forward(recon_input)

    keypoints_out = (out_recon['keypoints']* 0.5 + 0.5)
    keypoints = keypoints_out.sigmoid()

    return keypoints_out
