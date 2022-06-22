from torch import nn
import timm
import segmentation_models_pytorch as smp

class Seg(nn.Module):
    def __init__(
        self,
        decoder: str = 'unet',
        encoder_name: str = 'efficientnet-b0',
        encoder_weights: str = 'imagenet'
        classes: int = 3,
        
    ):
        super().__init__()
         if decoder.lower() == 'unet':
            self.model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                            in_channels=in_channels, classes=classes, activation=None)
        elif decoder.lower() == 'fpn':
            self.model = smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights,
                            in_channels=in_channels, classes=classes, activation=None)
        elif decoder.lower() == 'unetplusplus':
            self.model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                    in_channels=in_channels, classes=classes, activation=None)
        elif decoder.lower() == 'linknet':
            self.model = smp.Linknet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                in_channels=in_channels, classes=classes, activation=None)
        elif decoder.lower() == 'deeplabv3':
            self.model = smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                in_channels=in_channels, classes=classes, activation=None)
        elif decoder.lower() == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                    in_channels=in_channels, classes=classes, activation=None)
        elif decoder.lower() == 'pspnet':
            self.model = smp.PSPNet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                            in_channels=in_channels, classes=classes, activation=None)
        
        
    def forward(self, x):
        
        return self.model(x)
