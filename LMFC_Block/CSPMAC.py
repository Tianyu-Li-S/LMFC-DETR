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

__all__ = ['CSPMAC']

class MAC(nn.Module):
    def __init__(self, inc, ratio=[1, 2, 3]) -> None:
        super().__init__()
        
        self.conv1 = Conv(inc, inc, k=3, d=ratio[0])
        self.conv2 = Conv(inc, inc // 2, k=3, d=ratio[1])
        self.conv3 = Conv(inc, inc // 2, k=3, d=ratio[2])
        self.conv4 = Conv(inc * 2, inc, k=1)
    
    def forward(self, x):
        return self.conv4(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))
#################################################################################################################
class CSPMAC(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = MAC(c_)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
