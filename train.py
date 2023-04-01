#!/usr/bin/env python

import argparse

from utils.preprocessing import ForestryDataset
from utils.models import Models

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


parser = argparse.ArgumentParser(description="PyTorch BIP Segmentation")
parser.add_argument("-data", "--path", metavar="DIR", default="../../datasets/treecover_segmentation_aerial_goettingen/",
                    help="Path to the dataset")
parser.add_argument("-encoder", "--encoder", default="resnet50",
                    help="Encoder method to be used : ['resnet18', 'resnet34','resnet50', 'resnet101', 'resnet152','mobilenet_v2'] ")
parser.add_argument("-decoder", "--method", default="deeplabv3plus",
                    help="Choose ['unet','deeplabv3','deeplabv3plus']")
parser.add_argument("-lr", default=0.00001, type=float, 
                    help="Learning rate of the model")
parser.add_argument("-input", "--image_size", default=1024, type=int, 
                    help="Input image size, default = 1024x1024")
parser.add_argument("-epochs", default=80, type=int, 
                    help="Number of epochs to train the model" )
parser.add_argument("-wd", default=0.001, type=float, 
                    help = "Weight decay for L2 regularization")
parser.add_argument("-early_stop", action='store_true', default = False,
                    help="Early stopping if the loss stops decreasing for certain epochs")

def main():
    args = parser.parse_args()
    test_transformation = transforms.Compose([
            transforms.ToTensor(),
           # transforms.Resize(size=(512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    trainData = ForestryDataset(args.path, size = args.image_size, train = True, transform = None)
    testData = ForestryDataset(args.path, size = args.image_size, train = False, transform = test_transformation)

    # Train and test dataloaders
    train_dataloader = DataLoader(trainData, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(testData, batch_size=1, shuffle=False)

    # Load the model
    model_obj = Models(args.method, args.encoder)
    model = model_obj.getModel()

    # Set device: `cuda` or `cpu`
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # define loss function
    lossFunction = smp.utils.losses.DiceLoss()
    # define metrics
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    # define optimizer
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.lr,weight_decay=args.wd)])

    # Get train and test functions 
    train_epoch = model_obj.getTrainFunction(model, lossFunction, metrics, optimizer, device, True)
    test_epoch = model_obj.getValidFunction(model, lossFunction, metrics, device, True)


if __name__ == "__main__":
    main()