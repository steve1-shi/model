import math
import torch
import torch.nn as nn 
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import numpy as np



__all__ = ['starnet_s050', 'starnet_s100', 'starnet_s150', 'starnet_s1', 'starnet_s2', 'starnet_s3', 'starnet_s4']

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}

def trunc_normal_(tensor, mean=0., std=1.):
    
    with torch.no_grad():
        tensor.normal_(mean, std)
        while True:
            invalid = (tensor < mean - 2*std) | (tensor > mean + 2*std)
            if not invalid.any():
                break
            tensor[invalid] = tensor[invalid].normal_(mean, std)
        return tensor
    
# -----------------------------
# DropPath
# -----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# -----------------------------
# ----------------------- origin BNand BLock------
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x
# -----------------------------
# FlexConv
# -----------------------------
class FlexConv(nn.Module):
    """Pointwise / Depthwise """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, with_bn=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if with_bn else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))
# -----------------------------
# MDConv
# -----------------------------
class MDConv(nn.Module):
    """Multi-Dilation DWConv with concat + GN + SiLU + projection"""
    def __init__(self, dim, kernel_size=3, dilations=(1,2), gn_groups=8):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, stride=1,
                      padding=d*(kernel_size//2), dilation=d,
                      groups=dim, bias=False)
            for d in dilations
        ])
        self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=dim*len(dilations))
        self.proj = nn.Conv2d(dim*len(dilations), dim, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        x = torch.cat(outs, dim=1)
        x = self.act(self.gn(x))
        return self.proj(x)
# -----------------------------
# ---- replace ECA with this ----
class ECA(nn.Module):
    """Adaptive ECA: kernel size auto-scales with feature level"""
    def __init__(self, gamma=2, b=1, min_k=3, max_k=9):
        super().__init__()
        self.gamma = gamma
        self.b = b
        self.min_k = min_k
        self.max_k = max_k

    def _get_k(self, C):
        
        t = int(abs((math.log2(C) + self.b) / self.gamma))
        k = t if (t % 2) else t + 1
        return int(np.clip(k, self.min_k, self.max_k))

    def forward(self, x):
        B, C, _, _ = x.size()
        k = self._get_k(C)
        y = F.adaptive_avg_pool2d(x, 1).view(B, 1, C)
        conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False).to(x.device).to(x.dtype)
        y = conv(y)
        y = torch.sigmoid(y).view(B, C, 1, 1)
        return x * y
# -------------------------------

# BlockMD
# -----------------------------
class BlockMD(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., conv_module=FlexConv, use_add_gate=False):
        super().__init__()
        self.use_add_gate = use_add_gate
        self.dwconv = conv_module(dim, dim, 7, 1, groups=dim, with_bn=True)
        self.f1 = conv_module(dim, mlp_ratio*dim, 1, groups=1, with_bn=False)
        self.f2 = conv_module(dim, mlp_ratio*dim, 1, groups=1, with_bn=False)
        self.g  = conv_module(mlp_ratio*dim, dim, 1, groups=1, with_bn=True)
        self.mdconv = MDConv(dim, kernel_size=3, dilations=(1,2))
        self.eca = ECA()
        self.act = nn.SiLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        x = self.mdconv(x)
        x = self.eca(x)
        x = input + self.drop_path(x)
        return x
# -----------------------------
# StarNet
# -----------------------------
class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=(3,3,12,5), mlp_ratio=4, drop_path_rate=0.015, conv_module=FlexConv, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.base_dim = base_dim
        self.in_channel = base_dim

        self.stem = nn.Sequential(
            conv_module(3, base_dim//2, 3, stride=2),
            nn.SiLU(),
            conv_module(base_dim//2, base_dim, 3, stride=1),
            nn.SiLU()
        )

        # stochastic depth
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * (2 ** i_layer)
            down_sampler = conv_module(self.in_channel, embed_dim, 3, stride=2)
            self.in_channel = embed_dim
            blocks = [BlockMD(self.in_channel, mlp_ratio, dpr[cur + i], conv_module=conv_module) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.channel = [base_dim] + [base_dim*(2**i) for i in range(len(depths))]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            try:
                trunc_normal_(m.weight, std=.02)
            except Exception:
                nn.init.normal_(m.weight, mean=0., std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except Exception:
                pass

    def forward(self, x):
        features = []
        x = self.stem(x)
        features.append(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()



def starnet_s1(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model



def starnet_s2(pretrained=False, **kwargs):
    model = StarNet(32, [1, 2, 6, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model



def starnet_s3(pretrained=False, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model



def starnet_s4(pretrained=False, **kwargs):
    model = StarNet(32, [3, 3, 12, 5], **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


# very small networks #

def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)



def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)



def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)

