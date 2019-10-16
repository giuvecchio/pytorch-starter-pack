import torch
from torch import nn


class SELayer(nn.Module):
    """An unofficial implementation of SE block, proposed in 
    `Squeeze-and-Excitation Networks` by Hu et al. 
    (https://arxiv.org/abs/1709.01507).

    Adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.

    Args:
        in_planes (int): Number of feature maps in the input tensor
        reduction (int): Features reduction factor

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`
    """

    def __init__(self, in_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
