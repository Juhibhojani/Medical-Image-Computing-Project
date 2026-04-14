import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# returns a basics convolution block with conv2D, followed by batch norm and then GELU.
def conv_3x3_bn(inp, oup, kernel_size, downsample=False):
    stride = 1 if downsample == False else 2 # you can change the stride value in the stem block to 4 as discussed in the paper in the ablations
    padding_size = kernel_size//2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding_size, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    """
        Function to apply normalization before the given function.
    """
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# channel attention 
class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        # takes pooling on avg across all height and width and only keeps number of channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # squeeze
        # excitation module
        self.fc = nn.Sequential(
            # increases the number of channels an then decreases it
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# channel and spatial attention
class CBAM(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.spatial_attention=SpatialAttention(7)
        self.channel_attention= ChannelAttention(inp)
    def forward(self, x):
        residual=x
        x=self.channel_attention(x)*x
        x=self.spatial_attention(x)*x
        x=x+residual
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # applying a 7*7 convolution for each H*W to find its' importance wrt its neighbours
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # additive attention block
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.GELU(), 
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # same fc applied to it and then concantenates features
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class MBConv(nn.Module):
    """
      Uses SE module which includes channel attention only
      Args:
      inp (int): Input dimension.
      oup (int): Output dimension.
      kernel_size (int): Size of kernel. Default: 3
      downsample (bool): Boolean to include stride = 2 for downsampling or not. It is set to true in the first block of each stage.
      expansion (int): Expansion ratio. Default: 4

    """
    def __init__(self, inp, oup, kernel_size ,downsample=False, expansion=4):
        super().__init__()
        # we will use stride 2 in case I want to downsample
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        padding_size = kernel_size//2

        # increasing hidden dim by 4 - happens inside MBConv layer
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # as we have no expansion then no need of 1*1 convolution
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          padding_size, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                # mapping back to output number of channels
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # 1*1 conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                # this is depth wise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 1, padding_size,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                # project back to output
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class MBCSA(nn.Module):
    """
      Uses CBAM module which includes channel and spatial attentions
      Args:
      inp (int): Input dimension.
      oup (int): Output dimension.
      kernel_size (int): Size of kernel. Default: 3
      downsample (bool): Boolean to include stride = 2 for downsampling or not. It is set to true in the first block of each stage.
      expansion (int): Expansion ratio. Default: 4

    """
    def __init__(self, inp, oup, kernel_size ,downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        padding_size = kernel_size//2

        hidden_dim = int(inp * expansion)

        # TODO: understand this?
        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                # depth wise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          padding_size, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 1, padding_size,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                CBAM(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)
