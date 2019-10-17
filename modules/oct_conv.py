import torch
import torch.nn as nn


class OctaveConv(nn.Module):
    """An unofficial implementation of Octave Convolution, proposed in 
    `Drop an Octave: Reducing Spatial Redundancy in 
    Convolutional Neural Networks with Octave Convolution` by Chen et al. at ICCV 2019
    (https://arxiv.org/abs/1904.05049).
    
    Distinctly process high and low frequency informations.

    Args:
        in_planes (int): Number of feature maps in the input tensor
        out_planes (int): Number of feature maps produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        alpha (float): Feature maps balancing factor. Default: 0.5. 
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        norm_layer (optional): If not ``None`` used for features normalization. Default: ``nn.BatchNorm2d``
        activation (optional): If not ``None`` applies a non linearity. Default: ``nn.ReLU``

    Shape:
        - Input: :math:`(N, C_{h_in}, H_{h_in}, W_{h_in}), (N, C_{l_in}, H_{l_in}, W_{l_in}))`
        - Output: :math:`(N, C_{h_out}, H_{h_out}, W_{h_out}), 
        (N, C_{l_out}, H_{l_out}, W_{l_out})` where 
        `C_{h_out} = out_planes * (1-alpha)` and `C_{l_out} = out_planes * alpha`
    """

    def __init__(self, in_planes, out_planes, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = nn.Conv2d(int(alpha * in_planes), int(alpha * out_planes),
                             kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = nn.Conv2d(int(alpha * in_planes), out_planes - int(alpha * out_planes),
                             kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = nn.Conv2d(in_planes - int(alpha * in_planes), int(alpha * out_planes),
                             kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(in_planes - int(alpha * in_planes),
                             out_planes - int(alpha * out_planes),
                             kernel_size, 1, padding, dilation, groups, bias)
        if norm_layer:
            self.norm_h = norm_layer(int(out_planes*(1-alpha)))
            self.norm_l = norm_layer(int(out_planes * alpha))

        self.activation = activation()

    def forward(self, x):
        x_h, x_l = x

        if self.stride == 2:
            x_h, x_l = self.h2g_pool(x_h), self.h2g_pool(x_l)

        x_h2l = self.h2g_pool(x_h)

        x_h2h = self.h2h(x_h)
        x_l2h = self.l2h(x_l)

        x_l2l = self.l2l(x_l)
        x_h2l = self.h2l(x_h2l)

        x_l2h = self.upsample(x_l2h)
        x_h = x_l2h + x_h2h
        x_l = x_h2l + x_l2l

        if self.norm_h and self.norm_l:
            x_h = self.norm_h(x_h)
            x_l = self.norm_l(x_l)

        if self.activation:
            x_h = self.activation(x_h)
            x_l = self.activation(x_l)

        return x_h, x_l


class FirstOctaveConv(nn.Module):
    """An unofficial implementation of Octave Convolution, proposed in 
    `Drop an Octave: Reducing Spatial Redundancy in 
    Convolutional Neural Networks with Octave Convolution` by Chen et al. at ICCV 2019
    (https://arxiv.org/abs/1904.05049).
    
    Distinctly process high and low frequency informations.
    FirstOctaveConv get a input tensor and return two separate tensors for high and low frequencies.

    Args:
        in_planes (int): Number of feature maps in the input tensor
        out_planes (int): Number of feature maps produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        alpha (float): Feature maps balancing factor. Default: 0.5. 
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        norm_layer (optional): If not ``None`` used for features normalization. Default: ``nn.BatchNorm2d``
        activation (optional): If not ``None`` applies a non linearity. Default: ``nn.ReLU``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{h_out}, H_{h_out}, W_{h_out}), 
        (N, C_{l_out}, H_{l_out}, W_{l_out})` where 
        `C_{h_out} = out_planes * (1-alpha)` and `C_{l_out} = out_planes * alpha`
    """
    def __init__(self, in_planes, out_planes, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = nn.Conv2d(in_planes, int(alpha * out_planes),
                             kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(in_planes, out_planes - int(alpha * out_planes),
                             kernel_size, 1, padding, dilation, groups, bias)

        if norm_layer:
            self.norm_h = norm_layer(int(out_planes*(1-alpha)))
            self.norm_l = norm_layer(int(out_planes * alpha))

        self.activation = activation()

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        x_h2l = self.h2g_pool(x)
        x_h = x
        x_h = self.h2h(x_h)
        x_l = self.h2l(x_h2l)

        if self.norm_h and self.norm_l:
            x_h = self.norm_h(x_h)
            x_l = self.norm_l(x_l)

        if self.activation:
            x_h = self.activation(x_h)
            x_l = self.activation(x_l)

        return x_h, x_l


class LastOctaveConv(nn.Module):
    r"""An unofficial implementation of Octave Convolution, proposed in 
    `Drop an Octave: Reducing Spatial Redundancy in 
    Convolutional Neural Networks with Octave Convolution` by Chen et al. at ICCV 2019
    (https://arxiv.org/abs/1904.05049).
    
    Distinctly process high and low frequency informations.
    LastOctaveConv get two separate tensors for high and low frequencies 
    and return a single output tensor.

    Args:
        in_planes (int): Number of feature maps in the input tensor
        out_planes (int): Number of feature maps produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        alpha (float): Feature maps balancing factor. Default: 0.5. 
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        norm_layer (optional): If not ``None`` used for features normalization. Default: ``nn.BatchNorm2d``
        activation (optional): If not ``None`` applies a non linearity. Default: ``nn.ReLU``

    Shape:
        - Input: :math:`(N, C_{h_in}, H_{h_in}, W_{h_in}), (N, C_{l_in}, H_{l_in}, W_{l_in}))`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out}) where 
        `C_{out} = out_planes`
    """
    def __init__(self, in_planes, out_planes, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = nn.Conv2d(int(alpha * in_planes), out_planes,
                             kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(in_planes - int(alpha * in_planes),
                             out_planes,
                             kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        if norm_layer:
            self.norm = norm_layer(int(out_planes))

        self.activation = activation()

    def forward(self, x):
        x_h, x_l = x

        if self.stride == 2:
            x_h, x_l = self.h2g_pool(x_h), self.h2g_pool(x_l)

        x_l2h = self.l2h(x_l)
        x_h2h = self.h2h(x_h)
        x_l2h = self.upsample(x_l2h)

        x_h = x_h2h + x_l2h

        if self.norm:
            x_h = self.norm(x_h)

        if self.activation:
            x_h = self.activation(x_h)

        return x_h


if __name__ == '__main__':
    x = torch.rand(1, 3, 64, 64)
    conv_1 = FirstOctaveConv(3, 64, (3, 3))
    conv_2 = OctaveConv(64, 64, (3, 3))
    conv_3 = LastOctaveConv(64, 3, (3, 3))
    out = conv_1(x)
    out = conv_2(out)
    out = conv_3(out)
