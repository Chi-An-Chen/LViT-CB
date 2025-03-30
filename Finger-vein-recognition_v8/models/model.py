import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from models.utils import ConvModule, LinearModule, PreNorm    

class Attention_CASViT(nn.Module):
    def __init__(self, c2, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.c2 = c2
        self.qkv = nn.Conv2d(c2, 3 * c2, 1, stride=1, padding=0, bias=attn_bias)

        # 合併 ChannelOperation 和 SpatialOperation
        self.spatial_conv = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2)
        self.spatial_bn = nn.BatchNorm2d(c2)
        self.spatial_activation = nn.SELU()
        self.spatial_out = nn.Conv2d(c2, 1, 1, 1, 0, bias=False)

        self.channel_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_fc = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)

        self.dwc = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2)
        self.proj = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2)
        self.proj_drop = nn.Dropout(proj_drop)

    def spatial_channel_attention(self, x):
        # 空間操作
        s = self.spatial_conv(x)
        s = self.spatial_bn(s)
        s = self.spatial_activation(s)
        s = self.spatial_out(s).sigmoid()

        # 通道操作
        c = self.channel_pool(x)
        c = self.channel_fc(c).sigmoid()

        return x * s * c

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.spatial_channel_attention(q)
        k = self.spatial_channel_attention(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


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
        self.fc1 = ConvModule(self.c2  , self.c2  , k=7, g=self.c2, mode="CONV-NORM")
        self.fc2 = ConvModule(self.c2  , self.c2*2, k=1, mode="CONV-NORM-ACTV")
        self.fc3 = ConvModule(self.c2*2, self.c2  , k=1, mode="CONV-NORM")

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

class CompetitiveBlock(nn.Module):
    def __init__(self, c2, n_competitor, weight=0.8):
        super(CompetitiveBlock, self).__init__()
        self.c2 = c2
        self.n_competitor = n_competitor

        self.argmax   = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)

        self.conv2nd = ConvModule(self.c2, self.c2, k=3, s=1, mode="CONV-NORM-ACTV")
        self.conv_up = ConvModule(self.c2*2, self.c2, k=1, s=1, mode="CONV-NORM-ACTV")
        
        self.w_channel = weight
        self.w_xy = (1-weight) / 2

    def forward(self, x):
        # 1-st order
        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        x_1  = self.w_channel * x1_1 + self.w_xy * (x1_2 + x1_3)

        # 2-nd order
        x    = self.conv2nd(x)
        x2_1 = self.argmax(x)
        x2_2 = self.argmax_x(x)
        x2_3 = self.argmax_y(x)
        x_2  = self.w_channel * x2_1 + self.w_xy * (x2_2 + x2_3)

        xx = torch.cat((x_1,x_2),dim=1)

        return self.conv_up(xx)

class BasicBlock(nn.Module):
    def __init__(self, dim, drop_path=0.2, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.use_layer_scale = use_layer_scale

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = PreNorm(Mlp(dim))
        self.atn = PreNorm(Attention_CASViT(dim))
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
        self.stem   = PatchEmbed(3, chs[0], k=3, s=1)
        self.stage1 = self.make_layer(BasicBlock, chs[0], blks[0], stride=2)
        self.stage2 = self.make_layer(BasicBlock, chs[1], blks[1], stride=2)
        self.stage3 = self.make_layer(BasicBlock, chs[2], blks[2], stride=2)
        self.stage4 = self.make_layer(BasicBlock, chs[3], blks[3], stride=2)

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
