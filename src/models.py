from resnet import ResNet9
from vgg import VGG
import torchvision.models as models
import torch.nn as nn
import torch
import os
import logging


def get_model(data, args):
    if data == 'cifar10':
        # model = ResNet9(3,num_classes=10, args=args)
        model = VGG('VGG9', num_classes=10)
        # Load pretrained weights if specified
        if hasattr(args, 'pretrained_path') and args.pretrained_path is not None and os.path.exists(args.pretrained_path):
            logging.info(f"Loading pretrained ResNet9 model from: {args.pretrained_path}")
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load weights with strict=False to allow partial loading (e.g., different num_classes)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logging.warning(f"Missing keys when loading pretrained model: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading pretrained model: {unexpected_keys}")
            logging.info("Pretrained ResNet9 model loaded successfully")
            
    elif data == 'cifar100':
        model = VGG('VGG9',num_classes=100)
        # Load pretrained weights if specified
        if hasattr(args, 'pretrained_path') and args.pretrained_path is not None and os.path.exists(args.pretrained_path):
            logging.info(f"Loading pretrained VGG9 model from: {args.pretrained_path}")
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load weights with strict=False to allow partial loading (e.g., different num_classes)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logging.warning(f"Missing keys when loading pretrained model: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading pretrained model: {unexpected_keys}")
            logging.info("Pretrained VGG9 model loaded successfully")
            
    elif data == 'tinyimagenet':
        model = get_resnet18_64x64()

    return model
         

def get_resnet18_64x64(num_classes=200):
    """
    Returns a ResNet-18 model modified for 64x64 input images and a specific number of output classes.

    Args:
        num_classes (int): The number of output classes. Default is 200.

    Returns:
        model (torch.nn.Module): Modified ResNet-18 model.
    """
    # Load the ResNet-18 model pretrained on ImageNet
    model = models.resnet18(pretrained=False)

    # Modify the first convolutional layer to handle 64x64 images
    model.conv1 = nn.Conv2d(
        in_channels=3, 
        out_channels=64, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    
    # Adjust the max pooling layer to fit smaller image sizes
    model.maxpool = nn.Identity()

    # Modify the fully connected layer to output num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model