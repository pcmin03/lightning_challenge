from torch import nn
import timm

class Resnet(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = timm.create_model('resnet18',pretrained=True,num_classes=1)
        self.m = nn.Sigmoid()
        
    def forward(self, x):
        
        return self.m(self.model(x)).squeeze()
