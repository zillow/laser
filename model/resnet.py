import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import *
from torchvision.models._utils import IntermediateLayerGetter


class ResNet50(nn.Module):
    def __init__(self, V=16, D=128):
        super(ResNet50, self).__init__()
        self.V = V
        self.D = D
        backbone_raw = resnet50(
            pretrained=True, replace_stride_with_dilation=[False, False, False]
        )
        self.backbone = IntermediateLayerGetter(
            backbone_raw, return_layers={"layer4": "feat"}
        )
        self.fc = nn.Linear(2048, D)

    def forward(self, x, V=None):
        x = self.backbone(x)["feat"]  # N,D,fH,fW
        x = x.mean(dim=2)  # N,D,fW
        if V is None:
            V = self.V
        x = F.adaptive_avg_pool1d(x, V)  # N,D,V
        x = x.reshape(-1, 2048, V).permute(0, 2, 1)  # N,V,D
        x = self.fc(x)
        return x
