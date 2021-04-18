import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split

from PIL import Image
import numpy as np

import pandas as pd

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import albumentations as A
from albumentations.pytorch.transforms import ToTensor

class KeyDataset(object):
  def __init__(self, df, A_transforms):
    self.df = df
    self.transform = A_transforms
    super().__init__()

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, idx):
    # load images 
    image_id = self.df.iloc[idx, 0]
    image = Image.open(os.path.join("../Keypoint/train_imgs/", image_id)).convert('RGB')
    image = np.array(image)

    # load keypoints
    keypoints = self.df.iloc[idx, 1:].values.reshape(-1, 2)

    # load labels
    labels = np.array([1])

    # load boxes
    x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
    x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
    boxes = np.array([[x1, y1, x2, y2]])

    targets ={
    'image': image,
    'bboxes': boxes,
    'labels': labels,
    'keypoints': keypoints}

    if self.transform is not None:
      targets = self.transform(**targets)

    image = targets['image']

    target = {
    'labels': torch.as_tensor(targets['labels'], dtype=torch.int64),
    'boxes': torch.as_tensor(targets['bboxes'], dtype=torch.float32),
    'keypoints': torch.as_tensor(
      np.concatenate([targets['keypoints'], np.ones((24, 1))], axis=1)[np.newaxis], dtype=torch.float32
    )
    }

    return image, target

A_transforms = {
    'train':
        A.Compose([
    A.Resize(224, 224, always_apply=True),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.OneOf([A.HorizontalFlip(p=1),
                     A.RandomRotate90(p=1),
                     A.VerticalFlip(p=1),
                    
            ], p=0.5),
            
    ToTensor()
],  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True)
),
    
    'val':
        A.Compose([
            A.Resize(224, 224, always_apply=True),
             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
             ToTensor()
                                    
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, angle_in_degrees=True),
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
}



def collate_fn(batch: torch.Tensor):
    return tuple(zip(*batch))
    
def get_loader(config):
  df = pd.read_csv("../Keypoint/train_df.csv")
  # lst = [317, 869, 873, 877, 911, 1559, 1560, 1562, 1566, 1575, 1577, 1578, 1582, 1606, 1607, 1622, 1623, 1624, 1625, 1629, 3968, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194]
  # df = df.drop(lst).reset_index(drop = True)


  if config.train_ratio == 1.0:
    
    train_loader = DataLoader(dataset = KeyDataset(df, A_transforms["train"]), batch_size = config.batch_size, shuffle=True,
                             num_workers=8, collate_fn=collate_fn)
    return train_loader
    
  if config.train_ratio < 1.0:
    train_cnt = int(df.shape[0] * config.train_ratio)
    train, valid = train_test_split(df, train_size = train_cnt)
    train_loader = DataLoader(dataset = KeyDataset(train, A_transforms["train"]), batch_size = config.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset = KeyDataset(valid, A_transforms["val"]), batch_size = config.batch_size, shuffle = False,
                              num_workers=4, collate_fn=collate_fn)

    return train_loader, valid_loader

  