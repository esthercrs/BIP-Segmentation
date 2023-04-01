import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.train import TrainEpoch
from segmentation_models_pytorch.utils.train import ValidEpoch

class Models:
    """ Class to get models with their pretrained weights for segmentation
    using https://github.com/qubvel/segmentation_models.pytorch library
    """

    def __init__(self, model_name, encoder, output_classes=1, encoder_weights="imagenet", activation="sigmoid") -> None:
        
        if model_name == "unet":
            self.model = smp.Unet(
            encoder_name = encoder, 
            encoder_weights = encoder_weights, 
            classes = output_classes, 
            activation = activation)
       
        elif model_name == "deeplabv3plus":
            self.model = smp.DeepLabV3Plus(
            encoder_name = encoder, 
            encoder_weights = encoder_weights, 
            classes = output_classes, 
            activation = activation)
        
        elif model_name == "deeplabv3":
            self.model = smp.DeepLabV3(
            encoder_name = encoder, 
            encoder_weights = encoder_weights, 
            classes = output_classes, 
            activation = activation)
    
    """Return the model with pretrained weights"""
    def getModel(self):
        return self.model

    """Return the SMP train function"""
    def getTrainFunction(self, model, loss_function, metrics, optimizer, device, verbose):
        
        return TrainEpoch(model, loss=loss_function, metrics=metrics, 
                        optimizer=optimizer, device=device, verbose=True)
    
    """Return the SMP valid function"""
    def getValidFunction(self, model, loss_function, metrics, device, verbose):

        return ValidEpoch(model, loss=loss_function, metrics=metrics, 
                        device=device, verbose=True)