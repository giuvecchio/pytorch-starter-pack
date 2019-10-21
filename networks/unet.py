import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
        An unofficial implementation of U-Net, proposed in 
        `U-Net: Convolutional Networks for Biomedical Image Segmentation`
        by Ronneberger et al. at MICCAI 2015
        (https://arxiv.org/abs/1505.04597).

        The network architecture is based on the so called skip connections.
        It consists of a contracting path and an expansive path. 
        The contracting path follows the typical architecture of a convolutional network.
        Every step in the expansive path consists of an upsampling of the
        feature map followed by a 2x2 convolution (“up-convolution”) that halves the
        number of feature channels, a concatenation with the correspondingly cropped
        feature map from the contracting path (skip connection).

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            depth (int): Depth of the network
            filters_growth_rate (int): Used to compute filters number for each layer.
                                       Number of filters is computed as 2 ** (filters_growth_rate + layer_idx).
            padding (bool): If True, apply padding such that the input shape is the same as the output.
            norm_layer (bool): If not ``None`` used for features normalization. Default: ``nn.BatchNorm2d``
            up_mode (str): Upscale mode. One of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
    """

    def __init__(self, in_channels=1, out_channels=2, depth=5, filters_growth_rate=6, padding=True, norm_layer=nn.BatchNorm2d, up_mode='upconv'):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        # Downsampling path
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (filters_growth_rate + i),
                              padding, norm_layer)
            )
            prev_channels = 2 ** (filters_growth_rate + i)

        # Upsampling path
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (filters_growth_rate + i),
                            up_mode, padding, norm_layer)
            )
            prev_channels = 2 ** (filters_growth_rate + i)

        # Final conv
        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, padding, norm_layer=None):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_planes, out_planes,
                               kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if norm_layer:
            block.append(norm_layer(out_planes))

        block.append(nn.Conv2d(out_planes, out_planes,
                               kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if norm_layer:
            block.append(norm_layer(out_planes))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, up_mode, padding, norm_layer=None):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_planes, out_planes, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(
            in_planes, out_planes, padding, norm_layer)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


if __name__ == '__main__':
    net = UNet(in_channels=3, out_channels=3)
    x = torch.rand(1, 3, 64, 64)
    out = net(x)
    print(out.size())
