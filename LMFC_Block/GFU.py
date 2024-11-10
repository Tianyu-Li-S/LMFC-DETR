import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from einops import rearrange
from collections import OrderedDict
from timm.layers import trunc_normal_
from timm.layers import DropPath

__all__ = ['GFU']

class GFU(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(inc, inc, 1),
            nn.Hardsigmoid()
        )
        
        self.conv = Conv(inc, inc, k=1)
        self.up_branch1 = ConvTranspose(inc, inc // 2, 2, 2)
        self.up_branch2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc, inc // 2, k=1)
        )
        
    def forward(self, x):
        channel_gate = self.gate(self.globalpool(x))
        x_up = torch.cat([self.up_branch1(x), self.up_branch2(x)], dim=1) * channel_gate
        output = self.conv(x_up)
        return output