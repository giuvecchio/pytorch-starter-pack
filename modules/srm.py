import torch
from torch import nn


class SRMLayer(nn.Module):
    """An unofficial implementation of SRM block, proposed in 
    `SRM: A Style-based Recalibration Module for Convolutional Neural Networks` by Lee et al. 
    (https://arxiv.org/abs/1903.10829).

    Adaptively recalibrates intermediate feature maps by exploiting their styles.

    Args:
        in_planes (int): Number of feature maps in the input tensor

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`
    """
    def __init__(self, in_planes):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(in_planes, in_planes, kernel_size=2, bias=False,
                             groups=in_planes)
        self.bn = nn.BatchNorm1d(in_planes)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)