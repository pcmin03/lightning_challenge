from torch import nn
import timm
import segmentation_models_pytorch as smp
from monai.networks import nets
import monai
class Seg(nn.Module):

    def __init__(
        self,
        decoder: str = 'unet',
        encoder_name: str = 'efficientnet-b0',
        encoder_weights: str = 'imagenet',
        classes: int = 3,
        output_class : int = 4,
    ):
        if self.decoder == "unet":
            net =  monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        elif self.decoder == "attention_unet":
            net =  monai.networks.nets.AttentionUnet(
                spatial_dims=2,
                in_channels=3,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
            )
        elif self.decoder == "unetr":
            net =  monai.networks.nets.UNETR(
                in_channels=3,
                img_size=1024,
                out_channels=1,
                spatial_dims=2,
            )
        elif self.decoder == "swin_unetr":
            net =  monai.networks.nets.SwinUNETR(
                img_size=1024,
                in_channels=3,
                out_channels=1,
                spatial_dims=2,
            )

    def forward(self, x):
        
        return self.model(x)
