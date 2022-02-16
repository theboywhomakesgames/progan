from operator import index
from statistics import mode
import torch
from torch import nn
from torch.nn import functional as F

import common

class Discriminator(nn.Module):
    def __init__(self, feat_dim = 128):
        super().__init__()

        self.progression = nn.ModuleList([
            common.ConvBlock(feat_dim//4, feat_dim//4, 3, 1),
            common.ConvBlock(feat_dim//4, feat_dim//2, 3, 1),
            common.ConvBlock(feat_dim//2, feat_dim, 3, 1),
            common.ConvBlock(feat_dim, feat_dim, 3, 1),
            common.ConvBlock(feat_dim, feat_dim, 3, 1),
            common.ConvBlock(feat_dim, feat_dim, 3, 1),
            common.ConvBlock(feat_dim + 1, feat_dim, 3, 1, 4, 0)
        ])

        self.from_rgb = nn.ModuleList([
            common.ELRConv2d(3, feat_dim//4, 1),
            common.ELRConv2d(3, feat_dim//4, 1),
            common.ELRConv2d(3, feat_dim//2, 1),
            common.ELRConv2d(3, feat_dim, 1),
            common.ELRConv2d(3, feat_dim, 1),
            common.ELRConv2d(3, feat_dim, 1),
            common.ELRConv2d(3, feat_dim, 1)
        ])

        self.n_layer = len(self.progression)
        self.linear = common.ELRLinear(feat_dim, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            # if it's the first layer, turn input from rgb to feature maps
            if i == step:
                out = self.from_rgb[index](input)

            # add stats for variation control (Section 3 & Apendix 1.A)
            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            # apply CNN of this level
            out = self.progression[index](out)

            # if this is not the last layer (4x4), scale down
            if i > 0:
                # this is for feeding to the next prog layer
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                # this is for mixing with the last layer
                # when doing it progressively, all layers will be trained, but in each time we'll do this only once in the beginning
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        
        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)

        return out