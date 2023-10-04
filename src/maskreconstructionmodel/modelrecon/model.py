import importlib

import pytorch_lightning as pl
import torch.utils.data
import wandb

from src.maskreconstructionmodel.utils.loss import VGGPerceptualLoss
from src.maskreconstructionmodel.visualization import *
from PIL import Image
import torch.nn as nn
import torch


# from models.segmentation import get_outputs, draw_segmentation_map, draw_binary_map
import torchvision


class distance_loss(torch.nn.Module):
    def __init__(self):
        super(distance_loss, self).__init__()

    def forward(self, points):
        # original points
        mean = torch.mean(points, dim=1).unsqueeze(1)
        distance = torch.norm(points - mean, p=2, dim=2)
        std = distance.std(axis=1)
        loss = torch.mean(std)

        return torch.exp(-loss)



class Model(pl.LightningModule):  # change nn.Module to pl.lightningModule
    def __init__(self, **kwargs):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.encoder = importlib.import_module('src.maskreconstructionmodel.modelrecon.' + self.hparams.encoder).Encoder(self.hparams)
        self.decoder = importlib.import_module('src.maskreconstructionmodel.modelrecon.' + self.hparams.decoder).Decoder(self.hparams)
        self.batch_size = self.hparams.batch_size
        # self.test_func = importlib.import_module('datasets.' + self.hparams.dataset).test_epoch_end
        
        self.vgg_loss = VGGPerceptualLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        self.vgg_loss.eval()
        inputs, labels = batch['img'], batch['label']

        outputs = self.decoder(self.encoder(batch))


        imgs = denormalize(batch['img']).cpu()
        recon_batch = self.decoder(self.encoder(batch))
        scaled_kp = recon_batch['keypoints'] * self.hparams.image_size / 2 + self.hparams.image_size / 2

        heatmap = recon_batch['heatmap'].cpu()
        heatmap_overlaid = torch.cat([heatmap] * 3, dim=1) / heatmap.max()
        heatmap_overlaid = torch.clamp(heatmap_overlaid + imgs * 0.5, min=0, max=1)

        per_loss = self.vgg_loss(outputs['img'], batch['img'])
        l1_loss = self.l1_loss(outputs['img'], batch['img'])
        loss = l1_loss + per_loss
        
        self.log('train_loss', loss)


        # self.logger.experiment.log({'generated': [wandb.Image(draw_img_grid(denormalize(ori_inputs).cpu()), caption='labels:{}'.format(labels.cpu())),
        #                                           wandb.Image(draw_img_grid(denormalize(recon_batch['damaged_img']).cpu()), caption='damaged'),
        #                                           wandb.Image(draw_img_grid(denormalize(batch['ori_img']).cpu()), caption='original_image'),
        #                                           wandb.Image(draw_img_grid(denormalize(recon_batch['img']).cpu()), caption='reconstructed'),
        #                                           wandb.Image(draw_img_grid(heatmap_overlaid.cpu()), caption='heatmap_overlaid'),
        #                                           wandb.Image(draw_kp_grid_unnorm(recon_batch['heatmap'], scaled_kp), caption='heatmap'),
        #                                           wandb.Image(wandb.Image(draw_kp_grid(denormalize(ori_inputs).cpu(), scaled_kp)), caption='keypoints')]})

        return loss

    def validation_step(self, batch, batch_idx):
        return batch

    def validation_epoch_end(self, outputs):
        inputs, labels = outputs[0]['img'], outputs[0]['label']
        ori_inputs = inputs.clone()
        ori_imgs = outputs[0]['ori_img']

        self.log("val_loss", -self.global_step)
        labels = outputs[0]['label']
        imgs = denormalize(outputs[0]['img']).cpu()
        recon_batch = self.decoder(self.encoder(outputs[0]))
        scaled_kp = recon_batch['keypoints'] * self.hparams.image_size / 2 + self.hparams.image_size / 2

        heatmap = recon_batch['heatmap'].cpu()
        heatmap_overlaid = torch.cat([heatmap] * 3, dim=1) / heatmap.max()
        heatmap_overlaid = torch.clamp(heatmap_overlaid + imgs * 0.5, min=0, max=1)


        self.logger.experiment.log({'generated': [wandb.Image(draw_img_grid(denormalize(ori_inputs).cpu()), caption='labels:{}'.format(labels.cpu())),
                                                  wandb.Image(draw_img_grid(denormalize(recon_batch['damaged_img']).cpu()), caption='damaged'),
                                                  wandb.Image(draw_img_grid(denormalize(outputs[0]['ori_img']).cpu()), caption='original input'),
                                                  wandb.Image(draw_img_grid(denormalize(recon_batch['img']).cpu()), caption='reconstructed'),
                                                  wandb.Image(draw_img_grid(heatmap_overlaid.cpu()), caption='heatmap_overlaid'),
                                                  wandb.Image(draw_kp_grid_unnorm(recon_batch['heatmap'], scaled_kp), caption='heatmap'),
                                                  wandb.Image(wandb.Image(draw_kp_grid(denormalize(ori_inputs).cpu(), scaled_kp)), caption='keypoints')]})
                                                #   wandb.Image(draw_matrix(recon_batch['skeleton_scalar_matrix'].detach().cpu().numpy()), caption='skeleton_scalar')]})

    def test_step(self, batch, dataloader_idx=0):
        # outputs = self.decoder(self.encoder(batch))
        inputs, labels = batch['img'], batch['label']
        ori_inputs = inputs.clone()
        imgs = denormalize(batch['img']).cpu()
        imgs = denormalize(batch['img']).cpu()
        recon_batch = self.decoder(self.encoder(batch))
        scaled_kp = recon_batch['keypoints'] * self.hparams.image_size / 2 + self.hparams.image_size / 2

        heatmap = recon_batch['heatmap'].cpu()
        heatmap_overlaid = torch.cat([heatmap] * 3, dim=1) / heatmap.max()
        heatmap_overlaid = torch.clamp(heatmap_overlaid + imgs * 0.5, min=0, max=1)

        
        self.logger.experiment.log({'generated': [wandb.Image(draw_img_grid(denormalize(ori_inputs).cpu()), caption='labels:{}'.format(labels.cpu())),
                                                  wandb.Image(draw_img_grid(denormalize(recon_batch['damaged_img']).cpu()), caption='damaged'),
                                                  wandb.Image(draw_img_grid(denormalize(batch['ori_img']).cpu()), caption='original_image'),
                                                  wandb.Image(draw_img_grid(denormalize(recon_batch['img']).cpu()), caption='reconstructed'),
                                                  wandb.Image(draw_img_grid(heatmap_overlaid.cpu()), caption='heatmap_overlaid'),
                                                  wandb.Image(draw_kp_grid_unnorm(recon_batch['heatmap'], scaled_kp), caption='heatmap'),
                                                  wandb.Image(wandb.Image(draw_kp_grid(denormalize(ori_inputs).cpu(), scaled_kp)), caption='keypoints')]})


        return scaled_kp

    def test_epoch_end(self, outputs):

        inputs, labels = outputs[0]['img'], outputs[0]['label']
        ori_inputs = inputs.clone()
        ori_imgs = outputs[0]['ori_img']

        labels = outputs[0]['label']
        imgs = denormalize(outputs[0]['img']).cpu()
        recon_batch = self.decoder(self.encoder(outputs[0]))
        scaled_kp = recon_batch['keypoints'] * self.hparams.image_size / 2 + self.hparams.image_size / 2

        heatmap = recon_batch['heatmap'].cpu()
        heatmap_overlaid = torch.cat([heatmap] * 3, dim=1) / heatmap.max()
        heatmap_overlaid = torch.clamp(heatmap_overlaid + imgs * 0.5, min=0, max=1)


        self.logger.experiment.log({'generated': [wandb.Image(draw_img_grid(denormalize(ori_inputs).cpu()), caption='labels:{}'.format(labels.cpu())),
                                                  wandb.Image(draw_img_grid(denormalize(recon_batch['damaged_img']).cpu()), caption='damaged'),
                                                  wandb.Image(draw_img_grid(denormalize(outputs[0]['ori_img']).cpu()), caption='original input'),
                                                  wandb.Image(draw_img_grid(denormalize(recon_batch['img']).cpu()), caption='reconstructed'),
                                                  wandb.Image(draw_img_grid(heatmap_overlaid.cpu()), caption='heatmap_overlaid'),
                                                  wandb.Image(draw_kp_grid_unnorm(recon_batch['heatmap'], scaled_kp), caption='heatmap'),
                                                  wandb.Image(wandb.Image(draw_kp_grid(denormalize(ori_inputs).cpu(), scaled_kp)), caption='keypoints')]})
                                                #   wandb.Image(draw_matrix(recon_batch['skeleton_scalar_matrix'].detach().cpu().numpy()), caption='skeleton_scalar')]})


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-3)
        return optimizer
