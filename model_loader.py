# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:27:08 2020

@author: User
"""

import torch.nn as nn
from torchvision import models
from torch import nn
from torchvision.models import mobilenet_v2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers


def set_parameter_requires_grad(model, freeze):
    for param in model.parameters():
        param.requires_grad = not freeze


def get_model(config):

    model = None
    # input_size = 0

    if config.model_name == "resnet":
        """ Resnet34
        """
        model = models.resnet18(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, config.n_classes)
        # input_size = 224
    elif config.model_name == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes)
        # input_size = 224
    elif config.model_name == "vgg":
        """ VGG16_bn
        """
        model = models.vgg16_bn(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes)
        # input_size = 224
    elif config.model_name == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier.in_features
        model.classifier = nn.Linear(n_features, config.n_classes)
        # input_size = 224
        
    elif config.model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained = config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes)

        n_features

    elif config.model_name == "KeypointRCNN":
      backbone = models.mobilenet_v2(pretrained=True).features
      backbone.out_channels = 1280
      roi_pooler = MultiScaleRoIAlign(
          featmap_names=['0'],
          output_size=7,
          sampling_ratio=2
      )
      anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
      keypoint_roi_pooler = MultiScaleRoIAlign(
          featmap_names=['0'],
          output_size=14,
          sampling_ratio=2
      )

      model = KeypointRCNN(
          backbone, 
          num_classes=2,
          num_keypoints=24,
          box_roi_pool=roi_pooler,
          keypoint_roi_pool=keypoint_roi_pooler,rpn_anchor_generator=anchor_generator
      )

    elif config.model_name == "keypointrcnn_resnet50":
      model = models.detection.keypointrcnn_resnet50_fpn(pretrained=config.use_pretrained, progress=False)
      model.roi_heads.keypoint_predictor.kps_score_lowres = nn.ConvTranspose2d(512, 24, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    
    elif config.model_name == "keypointrcnn_resnet101":
      pretrained_backbone = True
      pretrained = False
      trainable_backbone_layers = None
      trainable_backbone_layers = _validate_trainable_layers(
              pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

      backbone = resnet_fpn_backbone('resnet101', pretrained_backbone, trainable_layers=trainable_backbone_layers)

      model = KeypointRCNN(
          backbone, 
          num_classes=2,
          num_keypoints=24)

    else:
        raise NotImplementedError('You need to specify model name.')

    return model






# class Image_regression(nn.Module):
    
#     def __init__(self):
#         super(Image_regression, self).__init__()
        
#         self.layer = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding = 1),
#             nn.BatchNorm2d(16),
            
#             nn.Conv2d(16, 16, 3, padding = 1),
#             nn.BatchNorm2d(16),
#             nn.MaxPool2d(2),
            
            
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(32, 64, 3, padding = 1),
#             nn.BatchNorm2d(64),
            
#             nn.Conv2d(64, 64, 3, padding = 1),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2)
#             )
        
#         self.fc = nn.Sequential(
#             nn.Linear(50176, 56),
#             nn.Linear(56, 1)
#         )
        
#     def forward(self, x):
#         out = self.layer(x)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         out = out.view(-1)
#         return out
