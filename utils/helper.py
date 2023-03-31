import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, verbose, delta, mode,model):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): Procedure for determining the best score.
            
            model : Will be used to store best models
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.mode = mode
        self.model=model

        if self.mode == 'min':
            self.criterion = np.less
            self.delta = - delta
            self.best_score = np.Inf

            self.vocab = {'score': 'loss', 'comportement': 'decreased'}

        elif self.mode == 'max':
            self.criterion = np.greater
            self.delta = delta
            self.best_score = np.NINF

            self.vocab = {'score': 'metric', 'comportement': 'increased'}

        else:
            raise ValueError(
                "mode only takes as value in input 'min' or 'max'")

    def __call__(self, score, model):
        """Determines if the score is the best and saves the model if so.
           Also manages early stopping.
        Arguments:
            score (float): Value of the metric or loss.
            model: Pytorch model
        """
        if np.isinf(self.best_score):
            self.best_score = score
            self.save_checkpoint(score, model)

        elif self.criterion(score, self.best_score + self.delta):

            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
            self.model=model
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        '''Saves the model when the score satisfies the criterion.'''
        if self.verbose:
            score_name = self.vocab['score']
            comportement = self.vocab['comportement']
            print(
                f'Validation {score_name} {comportement} ({self.best_score:.6f} --> {score:.6f}).  Saving model ...'
            )
        torch.save(model, './best_model.pth') 

"""Helper function to plot results"""
def plotResults(xList,y1List,y2List,y1Label,y2Label,xAxisLabel,yAxisLabel,title,save=False):
  sns.set()
  plt.plot(xList,y1List,label = y1Label)
  plt.plot(xList,y2List, label = y2Label)
  plt.legend()
  plt.xlabel(xAxisLabel)
  plt.ylabel(yAxisLabel)
  plt.title(title)
  if save:
    imageName = title.replace(" ", "_") + ".png"
    plt.savefig(imageName)
  plt.show()

""" Helper function for data visualization"""
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()
