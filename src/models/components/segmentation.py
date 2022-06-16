from torch import nn
import timm
import segmentation_models_pytorch as smp

class Seg(nn.Module):
    def __init__(
        self,
        decoder: str = 'unet',
        background: str = 'efficientnet-b0',
        num_classes: int = 3,
        
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=background,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,        # model output channels (number of classes in your dataset)
            activation=None,
        )
        
        
    def forward(self, x):
        
        return self.model(x)
