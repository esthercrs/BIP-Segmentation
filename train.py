#!/usr/bin/env python

import argparse

from utils.preprocessing import ForestryDataset
from utils.models import Models
from utils.helper import EarlyStopping, visualize

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
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

def visualizeResults(args, train_logs_list, test_logs_list):
    # Arrays to store the loss and iou score
    trainDiceLoss = []
    testDiceLoss = []
    trainIou = []
    testIou = []

    for log in train_logs_list:
        trainDiceLoss.append(log['dice_loss'])
        trainIou.append(log['iou_score'])

    for log in test_logs_list:
        testDiceLoss.append(log['dice_loss'])
        testIou.append(log['iou_score'])

    epochsList = np.arange(1,len(trainDiceLoss)+1,1)
    
    titleLoss = "Dice Loss for "+args.method+ "pretrained on" + args.encoder
    titleIoU = "IoU score for "+args.method+ "pretrained on" + args.encoder
    
    visualize(epochsList,trainDiceLoss,testDiceLoss,"Train Dice Loss","Test Dice Loss","Epochs","Dice Loss",titleLoss,save=True)
    visualize(epochsList,trainIou,testIou,"Train IoU score","Test IoU score","Epochs","IoU Score",titleIoU,save=True)


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

    # Main training block
    early_stopping = EarlyStopping(patience=10, verbose=True,delta=0.001,mode='min',model=model)
    best_iou_score = 0.0
    train_logs_list, test_logs_list = [], []

    for i in range(0, args.epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloader)
        test_logs = test_epoch.run(test_dataloader)
        train_logs_list.append(train_logs)
        test_logs_list.append(test_logs)

        early_stopping(test_logs['dice_loss'], model)
        if early_stopping.early_stop:
                print("Early stopping")
                break
        # Augmenting the same dataset again    
        trainData = ForestryDataset(args.path, size = args.image_size, train = True, transform = None)
        train_dataloader = DataLoader(trainData, batch_size=2, shuffle=True)

    visualizeResults(train_logs_list, test_logs_list)

if __name__ == "__main__":
    main()