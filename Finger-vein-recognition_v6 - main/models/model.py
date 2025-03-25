import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath
from models.utils import ConvModule, LinearModule, PreNorm

    

# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64):
#         super().__init__()
#         self.c2 = dim
#         inner_dim = dim_head *  heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.norm = nn.LayerNorm(dim)

#         self.attend = nn.Softmax(dim = -1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#         self.to_out = nn.Linear(inner_dim, dim, bias = False)


#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = rearrange(x, 'b c h w -> b (h w) c')

#         x = self.norm(x)
#         qkv = self.to_qkv(x).chunk(3, dim=-1)

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
    

class FocalModulation(nn.Module):
    def __init__(self, c2, proj_drop=0.):
        super().__init__()
        self.c2   = c2
        self.cv1  = ConvModule(c2        , c2*2+3, k=1, g=1    , mode="CONV-NORM")
        self.cv2  = ConvModule(c2//2     , c2//2 , k=3, g=c2//2, mode="CONV-NORM-ACTV")
        self.cv3  = ConvModule(c2//2     , c2//2 , k=3, g=c2//2, mode="CONV-NORM-ACTV")
        self.cv4  = ConvModule(c2+c2//2*3, c2    , k=1, g=1    , mode="CONV-NORM-ACTV")
        self.cv5  = ConvModule(c2        , c2    , k=1, g=1    , mode="CONV-NORM")
        self.drop = nn.Dropout(proj_drop)
        self.act  = nn.SiLU()

    def forward(self, x):
        b, c, h, w = x.shape
        q, ctx, gates = self.cv1(x).split([c, c, 3], dim=1)
        y1, y2 = ctx.chunk(2, dim=1)
        y3 = self.cv2(y2)
        y4 = self.cv3(y3)
        y5 = self.act(y4.mean((2, 3), keepdim=True))
        y  = torch.cat([y1, y2, y3 * gates[:, 0:1], y4 * gates[:, 1:2], y5 * gates[:, 2:3]], dim=1)
        y  = q * self.cv4(y)
        # 對modulate的特徵圖進行轉譯，並且隨機拋棄部分的特徵
        return self.cv5(y)


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, k=7, s=2):
        super().__init__()
        self.module = nn.Sequential(
            ConvModule(c1, c1, k, s=s, g=c1, mode="CONV-NORM"),
            ConvModule(c1, c2, 1, s=1, mode="CONV-NORM")
        )

    def forward(self, x):
        return self.module(x)
 

class Mlp(nn.Module):
    def __init__(self, c2):
        super().__init__()
        self.c2 = c2
        # self.cp = c2 // 4
        # self.cr = c2 - self.cp
        # self.fc1 = ConvModule(self.cp  , self.cp  , k=7, g=self.cp, mode="CONV-NORM")
        # self.fc2 = ConvModule(self.cp  , self.cp*4, k=1, mode="CONV-NORM-ACTV")
        # self.fc3 = ConvModule(self.cp*7, self.c2  , k=1, mode="CONV-NORM")
        self.fc1 = ConvModule(self.c2  , self.c2  , k=7, g=self.c2, mode="CONV-NORM")
        self.fc2 = ConvModule(self.c2  , self.c2*2, k=1, mode="CONV-NORM-ACTV")
        self.fc3 = ConvModule(self.c2*2, self.c2  , k=1, mode="CONV-NORM")

    def forward(self, x):
        # x1, x2 = x.split([self.cp, self.cr], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = torch.cat([x1, x2], dim=1)
        return self.fc3(x)

class CompetitiveBlock(nn.Module):
    def __init__(self, c2, n_competitor, weight=0.8):
        super(CompetitiveBlock, self).__init__()
        self.c2 = c2
        self.n_competitor = n_competitor

        self.argmax   = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        # PPU 
        # pad = ((stride - 1) + dilation * (n_competitor - 1)) // 2

        self.conv1st = ConvModule(self.c2, self.c2, k=n_competitor, s=1, mode="CONV-NORM")
        self.conv2nd = ConvModule(self.c2, self.c2, k=n_competitor, s=1, mode="CONV-NORM")
        
        self.conv1_1 = ConvModule(self.c2, self.c2, k=5, s=1, mode="CONV-NORM-ACTV")
        self.conv2_1 = ConvModule(self.c2, self.c2, k=5, s=1, mode="CONV-NORM-ACTV")

        self.conv_up = ConvModule(self.c2*2, self.c2, k=1, s=1, mode="CONV-NORM")
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.weight_chan = weight
        self.weight_spa = (1-weight) / 2

    def forward(self, x):
        B, C, H, W = x.shape
        # 1-st order
        x = self.conv1st(x)
        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        x_1  = self.weight_chan * x1_1 + self.weight_spa * (x1_2 + x1_3)

        x_1 = self.conv1_1(x_1)
        x_1 = self.maxpool(x_1)

        # 2-nd order
        x = self.conv2nd(x)
        x2_1 = self.argmax(x)
        x2_2 = self.argmax_x(x)
        x2_3 = self.argmax_y(x)
        x_2  = self.weight_chan * x2_1 + self.weight_spa * (x2_2 + x2_3)

        x_2  = self.conv2_1(x_2)
        x_2 = self.maxpool(x_2)

        xx = torch.cat((x_1,x_2),dim=1)
        xx = F.interpolate(xx, size=(H, W), mode='bilinear', align_corners=True) # bicubic

        return self.conv_up(xx)

class BasicBlock(nn.Module):
    def __init__(self, dim, drop_path=0.2, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.use_layer_scale = use_layer_scale

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = PreNorm(Mlp(dim))
        self.atn = PreNorm(FocalModulation(dim))
        self.comp = PreNorm(CompetitiveBlock(dim, n_competitor=9))

        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_3 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            # Competition
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.comp(x))
            # Focal Modulation
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.atn(x))
            # MLP
            x = x + self.drop_path(self.layer_scale_3.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        else:
            x = x + self.drop_path(self.comp(x))
            x = x + self.drop_path(self.atn(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class LightWeightedModel(nn.Module):
    def __init__(self, num_classes, chs=[32, 64, 96, 128], blks=[1, 1, 3, 1]):
        super().__init__()
        self.eps = 1e-5
        self.pre_dim = chs[0]
        self.stem   = PatchEmbed(3, chs[0], k=3, s=1)                                       # 112
        self.stage1 = self.make_layer(BasicBlock, chs[0], blks[0], stride=2)                # 56
        self.stage2 = self.make_layer(BasicBlock, chs[1], blks[1], stride=2)                # 28
        self.stage3 = self.make_layer(BasicBlock, chs[2], blks[2], stride=2)                # 14
        self.stage4 = self.make_layer(BasicBlock, chs[3], blks[3], stride=2)                # 7

        self.head   = LinearModule(chs[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.kernel = nn.Parameter(torch.FloatTensor(chs[3], num_classes))
    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if stride == 2:
                layers.append(PatchEmbed(self.pre_dim, channels, k=3, s=2))
            layers.append(block(channels))
        self.pre_dim = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        return self.head(x), x
