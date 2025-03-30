import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_


class PreNorm(nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.module = module
        self.norm = nn.BatchNorm2d(module.c2) 

    def forward(self, x):
        return self.module(self.norm(x))
    

class LinearModule(nn.Module):
    def __init__(self, dim_in, dim_out, std=0.02):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.norm = nn.BatchNorm1d(dim_in)
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        trunc_normal_(self.linear.weight, std=std)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        return self.linear(self.norm(x))


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k, s=1, g=1, d=1, mode="CONV-NORM-ACTV"):
        # ConvModule(self.c2  , self.c2  , k=7, g=self.c2, mode="CONV-NORM")
        p = ((s - 1) + d * (k - 1)) // 2
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        self.down = BlurPool(c1, stride=s) if s >= 2 else nn.Identity()
        self.conv = nn.Conv2d(c1, c2, k, 1, p, d, g, bias=False) if 'CONV' in mode else nn.Identity()
        self.norm = nn.BatchNorm2d(c2) if 'NORM' in mode else nn.Identity()
        self.actv = nn.SiLU() if 'ACTV' in mode else nn.Identity()
    
        trunc_normal_(self.conv.weight, std=.02)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
      
        if isinstance(self.norm, nn.BatchNorm2d):
            nn.init.constant_(self.norm.weight, 1)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.actv(x)
        return x
        

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def get_pad_layer(pad_type):
    if pad_type in ['refl','reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl','replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type=='zero':
        PadLayer = nn.ZeroPad2d
    elif pad_type == 'circular':
        PadLayer = CircularPad
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class CircularPad(nn.Module):
    def __init__(self, padding = (1, 1, 1, 1)):
        super().__init__()
        self.pad_sizes = padding
        
    def forward(self, x):
        return F.pad(x, pad = self.pad_sizes , mode = 'circular')
    

class Filter(nn.Module):
    def __init__(self, filt, channels, pad_type=None, pad_sizes=None, scale_l2=False, eps=1e-6):
        super(Filter, self).__init__()
        self.register_buffer('filt', filt[None, None, :, :].repeat((channels, 1, 1, 1)))
        if pad_sizes is not None:
            self.pad = get_pad_layer(pad_type)(pad_sizes)
        else:
            self.pad = None
        self.scale_l2 = scale_l2
        self.eps = eps

    def forward(self, x):
        if self.scale_l2:
            inp_norm = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        if self.pad is not None:
            x = self.pad(x)
        out = F.conv2d(x, self.filt, groups=x.shape[1])
        if self.scale_l2:
            out_norm = torch.norm(out, p=2, dim=(-1, -2), keepdim=True)
            out = out * (inp_norm / (out_norm + self.eps))
        return out


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='zero', filt_size=4, stride=2, pad_off=0, scale_l2=False, eps=1e-6):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        self.pad_type = pad_type
        self.scale_l2 = scale_l2
        self.eps = eps

        a = self.get_rect(self.filt_size)
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.filt = Filter(filt, channels, pad_type, self.pad_sizes, scale_l2)
        if self.filt_size == 1 and self.pad_off == 0:
            self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return self.filt(inp)[:, :, ::self.stride, ::self.stride]

    @staticmethod
    def get_rect(filt_size):
        if filt_size == 1:
            a = np.array([1., ])
        elif filt_size == 2:
            a = np.array([1., 1.])
        elif filt_size == 3:
            a = np.array([1., 2., 1.])
        elif filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        return a