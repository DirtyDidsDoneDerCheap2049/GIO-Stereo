import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalStatCalib(nn.Module):
    def __init__(self, channels, hidden=32, gamma_scale=0.10):
        super().__init__()
        self.gamma_scale = float(gamma_scale)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, 2 * channels, 1, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x, 1)
        h = self.act(self.fc1(g))
        gb = self.fc2(h)
        gamma, beta = gb.chunk(2, dim=1)
        return x * (1.0 + self.gamma_scale * gamma) + beta


class ConvINReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p, dilation=d, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class GlobalConsistencyContextNet(nn.Module):
    def __init__(self, in_channels=96, out_channels=96,
                 global_dilation=2, gsc_hidden=32, gsc_gamma_scale=0.10):
        super().__init__()
        C = out_channels

        self.trunk = nn.Sequential(
            ConvINReLU(in_channels, C, 3, 1, 1),
            ConvINReLU(C, C, 3, 1, 1),
        )

        self.global_branch = nn.Sequential(
            ConvINReLU(C, C, 3, p=global_dilation, d=global_dilation),
            nn.Conv2d(C, C, 1, bias=False),
        )
        nn.init.kaiming_normal_(self.global_branch[-1].weight, mode="fan_out", nonlinearity="relu")

        self.gsc1 = GlobalStatCalib(C, hidden=gsc_hidden, gamma_scale=gsc_gamma_scale)
        self.gsc2 = GlobalStatCalib(C, hidden=gsc_hidden, gamma_scale=gsc_gamma_scale)

        self.refine = ConvINReLU(C, C, 3, 1, 1)

    def forward(self, x):
        y = self.trunk(x)
        y = self.gsc1(y)

        g = self.global_branch(y)
        y = y + 0.5 * g

        y = self.gsc2(y)
        y = self.refine(y)
        return y
