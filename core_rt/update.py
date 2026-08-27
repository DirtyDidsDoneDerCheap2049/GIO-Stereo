import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_bilinear_interpolate(x, size=None, scale_factor=None):
    y = F.interpolate(
        x.float(), size=size, scale_factor=scale_factor,
        mode="bilinear", align_corners=False
    )
    return y.to(x.dtype)


def pool2x(x):
    return F.avg_pool2d(x, 2, 2)


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        mean = x32.mean(dim=(2, 3), keepdim=True)
        var = x32.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_hat = (x32 - mean) / (var + self.eps).sqrt()
        x_hat = x_hat.to(x.dtype)
        return x_hat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class FastMotionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = int(getattr(args, "hidden_dim", 96))
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) * 8
        mw = int(getattr(args, "motion_width", 64))

        self.convc1 = nn.Conv2d(cor_planes, mw, 1, bias=True)
        self.convc2 = nn.Conv2d(mw, mw, 3, padding=1, bias=True)
        self.convd1 = nn.Conv2d(1, mw, 3, padding=1, bias=True)
        self.convd2 = nn.Conv2d(mw, mw, 3, padding=1, bias=True)
        self.proj = nn.Conv2d(2 * mw, self.hidden_dim - 1, 3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, disp, corr):
        c = self.act(self.convc1(corr))
        c = self.act(self.convc2(c))
        d = self.act(self.convd1(disp))
        d = self.act(self.convd2(d))
        x = torch.cat([c, d], dim=1)
        x = self.act(self.proj(x))
        return torch.cat([x, disp], dim=1)


class TinySE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class FastResBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv3 = nn.Conv2d(C, C, 3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C, C, 1, bias=True)

    def forward(self, x):
        y = self.conv1(self.act(self.conv3(x)))
        return x + y


class FastDownBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv3 = nn.Conv2d(C, C, 3, stride=2, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C, C, 1, bias=True)
        self.short = nn.AvgPool2d(2, 2)

    def forward(self, x):
        y = self.conv1(self.act(self.conv3(x)))
        return y + self.short(x)


class DownBlock(nn.Module):
    def __init__(self, C, expand_ratio=1.25):
        super().__init__()
        exp = max(C, int(C * expand_ratio))
        self.conv = nn.Conv2d(C, C, 3, stride=2, padding=1, bias=True)
        self.norm = LayerNorm2d(C)
        self.pw1 = nn.Conv2d(C, exp, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv2d(exp, C, 1, bias=True)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        shortcut = self.pool(x)
        y = self.conv(x)
        y = self.norm(y)
        y = self.pw2(self.act(self.pw1(y)))
        return shortcut + y


class DilatedBlock(nn.Module):
    def __init__(self, C, expand_ratio=1.25, dilation=2):
        super().__init__()
        exp = max(C, int(C * expand_ratio))
        self.conv = nn.Conv2d(C, C, 3, padding=dilation, dilation=dilation, bias=True)
        self.norm = LayerNorm2d(C)
        self.pw1 = nn.Conv2d(C, exp, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv2d(exp, C, 1, bias=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.pw2(self.act(self.pw1(y)))
        return x + y


class DownBBlock(nn.Module):
    def __init__(self, C, expand_ratio=1.25):
        super().__init__()
        self.down = DownBlock(C, expand_ratio=expand_ratio)
        self.refine = DilatedBlock(C, expand_ratio=expand_ratio, dilation=2)

    def forward(self, x):
        return self.refine(self.down(x))


class MidResBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv3 = nn.Conv2d(C, C, 3, padding=1, bias=True)
        self.norm = LayerNorm2d(C)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C, C, 1, bias=True)

    def forward(self, x):
        y = self.conv3(x)
        y = self.norm(y)
        y = self.conv1(self.act(y))
        return x + y


class UpFuseLite(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * C, C, 1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, low, high):
        up = safe_bilinear_interpolate(low, size=high.shape[-2:])
        return self.fuse(torch.cat([up, high], dim=1))


class UpFuseNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv2d(2 * C, C, 1, bias=True)
        self.norm = LayerNorm2d(C)
        self.act = nn.ReLU(inplace=True)

    def forward(self, low, high):
        up = safe_bilinear_interpolate(low, size=high.shape[-2:])
        x = self.conv(torch.cat([up, high], dim=1))
        x = self.norm(x)
        x = self.act(x)
        return x


class DispHead(nn.Module):
    def __init__(self, in_channels, hidden=48):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True)
        )

    def forward(self, x):
        return self.head(x)


class MaskHeadFast(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.head(x)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        C = int(getattr(args, "hidden_dim", 96))
        expand_ratio = float(getattr(args, "update_expand_ratio", 1.25))

        self.encoder = FastMotionEncoder(args)
        self.inc_reduce = nn.Sequential(nn.Conv2d(2 * C, C, 1, bias=True), nn.ReLU(inplace=True))
        self.inc_att = TinySE(C, reduction=16)

        self.block_hi_a = nn.Sequential(
            nn.Conv2d(2 * C, C, 1, bias=True),
            nn.ReLU(inplace=True),
            FastResBlock(C)
        )
        self.block_down_a = nn.Sequential(
            nn.Conv2d(2 * C, C, 1, bias=True),
            nn.ReLU(inplace=True),
            FastDownBlock(C)
        )

        self.block_down_b = nn.Sequential(
            nn.Conv2d(2 * C, C, 1, bias=True),
            nn.ReLU(inplace=True),
            DownBBlock(C, expand_ratio=expand_ratio)
        )

        self.up_fuse_1 = UpFuseLite(C)
        self.up_fuse_2 = UpFuseNorm(C)

        self.block_hi_b = nn.Sequential(
            nn.Conv2d(2 * C, C, 1, bias=True),
            nn.ReLU(inplace=True),
            MidResBlock(C)
        )

        self.disp_head = DispHead(C, hidden=48)
        self.mask_head = MaskHeadFast(C, out_channels=32)
        self.register_buffer("step_scale", torch.tensor(1.0))
        self.final_gate = nn.Parameter(torch.tensor(0.60))

    def forward(self, net, context, geo_feat, disp):
        motion = self.encoder(disp, geo_feat)
        inc = self.inc_reduce(torch.cat([context, motion], dim=1))
        inc = self.inc_att(inc)

        x1 = self.block_hi_a(torch.cat([net, inc], dim=1))
        x2 = self.block_down_a(torch.cat([x1, inc], dim=1))

        inc_2x = pool2x(inc)
        x3 = self.block_down_b(torch.cat([x2, inc_2x], dim=1))

        y2 = self.up_fuse_1(x3, x2)
        y1 = self.up_fuse_2(y2, x1)

        new_net = self.block_hi_b(torch.cat([y1, inc], dim=1))
        g = torch.clamp(self.final_gate, 0.30, 0.70)
        net = g * new_net + (1.0 - g) * net

        delta_disp = self.disp_head(net) * self.step_scale
        mask_feat_4 = self.mask_head(net)
        return net, mask_feat_4, delta_disp
