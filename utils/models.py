import segmentation_models_pytorch as smp

class Models:
    """ Class to get models with their pretrained weights for segmentation
    using https://github.com/qubvel/segmentation_models.pytorch library
    """

    def __init__(self, model_name, encoder, encoder_weights, activation,output_classes) -> None:
        
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

    def getModel(self):
        return self.model