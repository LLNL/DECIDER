import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


import sys
sys.path.append("..")
sys.path.append("../models")

from models.mae import models_mae

import os
import numpy as np
from PIL import Image

class SAMBackbone(nn.Module):
    def __init__(self, model_name, checkpoint_path=None):
        super().__init__()
        """
        Args:
            model_name (str): Name of the model to load from the registry: [vit_h, vit_l, vit_b]
            checkpoint_path (str): Path to the checkpoint file
            device (str): Device to load the model on
        """
        if checkpoint_path is None:
            assert False, "Checkpoint path must be provided for {model_name}"
        elif not os.path.exists(checkpoint_path):
            assert False, f"Checkpoint path does not exist: {checkpoint_path}"

        try:
            self.model = sam_model_registry[model_name](checkpoint=checkpoint_path)

            # TODO: Fix this
            self.feature_dim = 4096
            # self.predictor = SamPredictor(sam)    
            self.image_encoder = self.model.image_encoder
            # Transform to resize the image to the longest side, add the preprocess that the model expects

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((1024, 1024)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])                
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            # Add a max pooling layer with stride 2 to reduce the dimensionality of the features
            self.pool = nn.MaxPool2d(kernel_size=16, stride=16)
        except Exception as e:
            assert False, f"Failed to load SAM model: {e}"
    
    def preprocess_pil(self, images):
        # check if images is a list, then preprocess each image
        if isinstance(images, list):
            images_torch = []
            for image in images:
                image_tensor = self.transform(image)
                images_torch.append(image_tensor)
            images_torch = torch.stack(images_torch).to(device) 
        else:
            images_torch = self.transform(images).unsqueeze(0).to(device)
        
        return images_torch
    @torch.no_grad()
    def forward(self, images):
        """
        Args:
            images (pil image(s) or torch.Tensor): Image(s) to extract features from:
                if pil image(s) are provided, they will be converted to torch.Tensor
                if torch.Tensor is provided, it must be of shape (N, C, H, W) the longer side must be 1024
        Returns:
            torch.Tensor: Features of the image(s) of shape (N, 256,64,64)
        """

        if not isinstance(images, torch.Tensor):
            images = self.preprocess_pil(images)
        else:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

        features = self.image_encoder(images)
        features = self.pool(features)
        features = features.view(features.shape[0], -1)
        return features

class MAEBackbone(nn.Module):
    def __init__(self, model_name, checkpoint_path=None):
        super().__init__()
        """
        Args:
            model_name (str): Name of the model to load from the registry: [vit_h, vit_l, vit_b]
            checkpoint_path (str): Path to the checkpoint file
            device (str): Device to load the model on
        """
        if checkpoint_path is None:
            assert False, "Checkpoint path must be provided for {model_name}"
        elif not os.path.exists(checkpoint_path):
            assert False, f"Checkpoint path does not exist: {checkpoint_path}"

        try:
            self.model = getattr(models_mae, model_name)()
            # load model
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            msg = self.model.load_state_dict(checkpoint['model'], strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(checkpoint_path, msg))
            # Transform to resize the image to the longest side, add the preprocess that the model expects
            # Link: https://github.com/facebookresearch/mae/blob/main/main_linprobe.py

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])                
            ])
            self.test_transform = transforms.Compose([
                            transforms.Resize(256, interpolation=3),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])                
                        ])

            self.feature_dim = 1024

        except Exception as e:
            assert False, f"Failed to load model: {e}"
    
    def preprocess_pil(self, images):
        # check if images is a list, then preprocess each image
        if isinstance(images, list):
            images_torch = []
            for image in images:
                image_tensor = self.transform(image)
                images_torch.append(image_tensor)

            images_torch = torch.stack(images_torch) 
        else:
            images_torch = self.transform(images).unsqueeze(0)
        
        return images_torch
    @torch.no_grad()
    def forward(self, images, decode=False):
        """
        Args:
            images (pil image(s) or torch.Tensor): Image(s) to extract features from:
                if pil image(s) are provided, they will be converted to torch.Tensor
        """
        if not isinstance(images, torch.Tensor):
            images = self.preprocess_pil(images)
        else:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
       
        features,_, ids_restore = self.model.forward_encoder(images, mask_ratio=0)

        if decode:
            pred = self.model.forward_decoder(features, ids_restore)
            return features[:,0], pred

        return features[:,0]

class DINOBackbone(nn.Module):
    def __init__(self, model_name, checkpoint_path=None):
        super().__init__()
        """
        Args:
            model_name (str): Name of the model to load from the registry: [vit_h, vit_l, vit_b]
            checkpoint_path (str): Path to the checkpoint file
        """

        try:
            self.model = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=True)
            # From: https://github.com/facebookresearch/dino/blob/main/eval_linear.py
            
            
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])                
            ])
            self.test_transform = transforms.Compose([
                            transforms.Resize(256, interpolation=3),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])                
                        ])
            self.feature_dim = 384

        except Exception as e:
            assert False, f"Failed to load model: {e}"
    
    def preprocess_pil(self, images, image_size=224, image_crop_size=224):
        # check if images is a list, then preprocess each image
        if isinstance(images, list):
            images_torch = []
            for image in images:
                image_tensor = self.transform(image)
                images_torch.append(image_tensor)

            images_torch = torch.stack(images_torch).to(device) 

        else:
            images_torch = self.transform(images).unsqueeze(0).to(device)
        
        return images_torch
    @torch.no_grad()
    def forward(self, images):
        """
        Args:
            images (pil image(s) or torch.Tensor): Image(s) to extract features from:
                if pil image(s) are provided, they will be converted to torch.Tensor
        """
        if not isinstance(images, torch.Tensor):
            images = self.preprocess_pil(images)
        else:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

        features = self.model(images)
        return features

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = SAMBackbone(model_name="vit_h", checkpoint_path="checkpoints/sam_vit_h_4b8939.pth").to(device)
    # mae = MAEBackbone(model_name="mae_vit_large_patch16", checkpoint_path='./checkpoints/mae_visualize_vit_large_ganloss.pth').to(device)
    # dino = DINOBackbone(model_name="dino_vits16", checkpoint_path=None).to(device)

    pil_image = Image.open("./data/domainnet_v1.0/real/toothpaste/real_318_000284.jpg")
    # pil_images = [pil_image, pil_image, pil_image, pil_image, pil_image, pil_image, pil_image, pil_image]
    pil_images = [pil_image]

    torch_images = torch.randn(4, 3, 224, 224).to(device)

    with torch.no_grad():
        features = sam(pil_images)
        # Flatten features using average pooling
        # features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

        # features = mae(torch_images)
        # features = dino(pil_images)

    print(features.shape)
