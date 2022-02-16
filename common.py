import torch
from torch import nn
from torch.nn import functional as F

import math

# ---------------------------------------------------------------------------- #
#                            Equalized Lerning Rate                            #
# ---------------------------------------------------------------------------- #
class ELR:
    # makes an ELR module
    @staticmethod
    def apply(module, name):
        elr = ELR(name)

        weights = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(
            name + "_",
            nn.Parameter(weights.data, requires_grad=True)
        )
        module.register_forward_pre_hook(elr)

        return elr

    def __init__(self, name):
        self.name = name

    # when the module is doing forward passing
    def __call__(self, module, input):
        usable_weights = self.compute_weights(module)
        setattr(module, self.name, usable_weights)
    
    # compute the ELR weights
    def compute_weights(self, module):
        original_weights = getattr(module, self.name + "_")
        fanin = original_weights.data.size(1) * original_weights.data[0][0].numel()

        return original_weights * math.sqrt(2 / fanin)

def equal_lr(module, name='weight'):
    ELR.apply(module, name)

    return module

# ---------------------------------------------------------------------------- #
#                          Pixel Normalization Module                          #
# ---------------------------------------------------------------------------- #
class PixelNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(
            torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8
        )

# ---------------------------------------------------------------------------- #
#                                ELRized Layers                                #
# ---------------------------------------------------------------------------- #
class ELRConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class ELRConvT2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class ELRLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        linear = nn.Linear(*args, **kwargs)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

# ---------------------------------------------------------------------------- #
#                                  Conv Block                                  #
# ---------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kern_size, pad, kern_size2=None, pad2=None, pixel_norm=True):
        super().__init__()

        if(pad2 is None):
            pad2 = pad

        if(kern_size2 is None):
            kern_size2 = kern_size

        seq = []
        seq.append(
            ELRConv2d(in_c, out_c, kern_size, padding=pad)
        )
        if pixel_norm:
            seq.append(PixelNormalizer())
        seq.append(nn.LeakyReLU(0.1))
        seq.append(ELRConv2d(out_c, out_c, kern_size2, padding=pad2))
        if pixel_norm:
            seq.append(PixelNormalizer())
        seq.append(nn.LeakyReLU(0.1))

        self.conv = nn.Sequential(*seq)

    def forward(self, input):
        out = self.conv(input)
        return out

def upscale(feat):
    return F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)