from matplotlib.colors import to_rgb
import torch
from torch import nn
from torch.nn import functional as F

import common

class Generator(nn.Module):
    def __init__(self, input_dim=128, in_c=128, pixel_norm=True, tanh=True):
        super().__init__()
        self.input_dim = input_dim
        self.tanh = tanh
        
        # input layer
        self.input_layer = nn.Sequential(
            common.ELRConvT2d(input_dim, in_c, 4),
            common.PixelNormalizer(),
            nn.LeakyReLU(0.1)
        )
        # state: in_c x 4 x 4 -> 128 x 4 x 4

        # progressive layers
        self.prog4 = common.ConvBlock(in_c, in_c, 3, 1, pixel_norm=pixel_norm)
        self.prog8 = common.ConvBlock(in_c, in_c, 3, 1, pixel_norm=pixel_norm)
        self.prog16 = common.ConvBlock(in_c, in_c, 3, 1, pixel_norm=pixel_norm)
        self.prog32 = common.ConvBlock(in_c, in_c, 3, 1, pixel_norm=pixel_norm)
        self.prog64 = common.ConvBlock(in_c, in_c//2, 3, 1, pixel_norm=pixel_norm)
        self.prog128 = common.ConvBlock(in_c//2, in_c//4, 3, 1, pixel_norm=pixel_norm)
        self.prog256 = common.ConvBlock(in_c//4, in_c//4, 3, 1, pixel_norm=pixel_norm)

        # to rgb layers
        self.to_rgb_4 = common.ELRConv2d(in_c, 3, 1)
        self.to_rgb_8 = common.ELRConv2d(in_c, 3, 1)
        self.to_rgb_16 = common.ELRConv2d(in_c, 3, 1)
        self.to_rgb_32 = common.ELRConv2d(in_c, 3, 1)
        self.to_rgb_64 = common.ELRConv2d(in_c//2, 3, 1)
        self.to_rgb_128 = common.ELRConv2d(in_c//4, 3, 1)
        self.to_rgb_256 = common.ELRConv2d(in_c//4, 3, 1)

        # max # of layers
        self.max_step = 6

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = common.upscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        
        if self.tanh:
            out = torch.tanh(out)

        return out

    def forward(self, input, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        # input noise 128 x 1 x 1 - input layer turns latent to 2d tensor
        out_4 = self.input_layer(input.view(-1, self.input_dim, 1, 1))
        # batchsize x 128 x 4 x 4
        # adds detail
        out_4 = self.prog4(out_4)
        # 128 x 4 x 4

        if step == 0:
            if self.tanh:
                return torch.tanh(self.to_rgb_4(out_4))
            else:
                self.to_rgb_4(out_4)

        # scale up
        out_8 = self.progress(
            out_4,
            self.prog8
        )
        # 128 x 8 x 8
        if step == 1:
            return self.output(
                out_4, out_8, self.to_rgb_4, self.to_rgb_8, alpha
            )

        # scale up
        out_16 = self.progress(
            out_8,
            self.prog16
        )
        # 128 x 16 x 16
        if step == 2:
            return self.output(
                out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha
            )

        # scale up
        out_32 = self.progress(
            out_16, 
            self.prog32
        )
        # 128 x 32 x 32
        if step == 3:
            return self.output(
                out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha
            )

        # scale up
        out_64 = self.progress(
            out_32,
            self.prog64
        )
        # 64 x 64 x 64
        if step == 4:
            return self.output(
                out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha
            )

        # scale up
        out_128 = self.progress(
            out_64,
            self.prog128
        )
        # 32 x 128 x 128
        if step == 5:
            return self.output(
                out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha
            )

        # scale up
        out_256 = self.progress(
            out_128,
            self.prog256
        )
        # 16 x 256 x 256
        return self.output(
            out_128,
            out_256,
            self.to_rgb_128,
            self.to_rgb_256,
            alpha
        )