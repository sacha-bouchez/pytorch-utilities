# Source - https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
# Posted by Poe Dator
# Retrieved 11/5/2025, License - CC-BY-SA 4.0

import torch.nn as nn

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
