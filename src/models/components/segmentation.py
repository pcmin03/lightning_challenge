from torch import nn
import timm
import segmentation_models_pytorch as smp
from monai.networks import nets

class Seg(nn.Module):
    def __init__(
        self,
        decoder: str = 'unet',
        encoder_name: str = 'efficientnet-b0',
        encoder_weights: str = 'imagenet',
        classes: int = 3,
        
    ):
        super().__init__()
        if decoder.lower() == 'unet':
            self.model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                            in_channels=classes, classes=3, activation=None)
        elif decoder.lower() == 'fpn':
            self.model = smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights,
                            in_channels=classes, classes=3, activation=None)
        elif decoder.lower() == 'unetplusplus':
            self.model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                    in_channels=classes, classes=3, activation=None)
        elif decoder.lower() == 'linknet':
            self.model = smp.Linknet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                in_channels=classes, classes=3, activation=None)
        elif decoder.lower() == 'deeplabv3':
            self.model = smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                in_channels=classes, classes=3, activation=None)
        elif decoder.lower() == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                    in_channels=classes, classes=3, activation=None)
        elif decoder.lower() == 'pspnet':
            self.model = smp.PSPNet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                            in_channels=classes, classes=3, activation=None)
        
    def forward(self, x):
        
        return self.model(x)

class Seg3d(nn.Module): 
    def __init__(
        self,
        decoder: str = 'unet', 
        encoder_name : str = '3d',
        classes: int = 3

    ):
        super().__init__()

        self.model = nets.UNet(
                    spatial_dims=classes,
                    in_channels=1,
                    out_channels=3,
                    channels=(32, 64, 128, 256, 512),
                    strides=(2,2,1,1),#(2, 2, 2, 2),
                    kernel_size=3,
                    up_kernel_size=3,
                    num_res_units=2,
                    act="PRELU",
                    norm="BATCH",
                    dropout=0.2,
                    bias=True,
                    dimensions=None,
                )
        
    def forward(self, x):
        return self.model(x)