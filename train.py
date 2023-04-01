#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="PyTorch BIP Segmentation")
parser.add_argument("-data", "--path", metavar="DIR", 
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
    

if __name__ == "__main__":
    main()