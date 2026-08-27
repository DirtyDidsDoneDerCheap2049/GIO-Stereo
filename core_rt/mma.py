import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_bilinear_interpolate(x, size=None, scale_factor=None):
    y = F.interpolate(x.float(), size=size, scale_factor=scale_factor,
                      mode="bilinear", align_corners=False)
    return y.to(x.dtype)


class BoundedAlpha(nn.Module):
    def __init__(self, init, lo=0.0, hi=0.35):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init)))
        self.lo = float(lo)
        self.hi = float(hi)

    def forward(self):
        return torch.clamp(self.alpha, self.lo, self.hi)


class LandmarkEpipolarAttn(nn.Module):
    def __init__(self, channels, d=16, num_landmarks=32):
        super().__init__()
        self.d = d
        self.m = num_landmarks

        self.to_q = nn.Conv2d(channels, d, 1, bias=False)
        self.to_k = nn.Conv2d(channels, d, 1, bias=False)
        self.to_v = nn.Conv2d(channels, d, 1, bias=False)
        self.proj = nn.Conv2d(d, channels, 1, bias=False)

        for m in [self.to_q, self.to_k, self.to_v, self.proj]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_landmarks(self, x_1d, m):
        BH, W, d = x_1d.shape
        if m >= W:
            return x_1d
        pad = (m - (W % m)) % m
        if pad > 0:
            x_1d = torch.cat([x_1d, x_1d[:, -1:, :].expand(BH, pad, d)], dim=1)
            W = W + pad
        s = W // m
        x_1d = x_1d.reshape(BH, m, s, d).mean(dim=2)
        return x_1d

    def _attend_one(self, q_in, kv_in):
        b, _, h, w = q_in.shape

        q = self.to_q(q_in)
        k = self.to_k(kv_in)
        v = self.to_v(kv_in)

        # (B*H, W, d)
        q = q.permute(0, 2, 3, 1).reshape(b * h, w, self.d)
        k = k.permute(0, 2, 3, 1).reshape(b * h, w, self.d)
        v = v.permute(0, 2, 3, 1).reshape(b * h, w, self.d)

        # landmarks: (B*H, m, d)
        km = self._make_landmarks(k, self.m)
        vm = self._make_landmarks(v, self.m)

        # scores: (B*H, W, m)
        scores = torch.bmm(q, km.transpose(1, 2)) / (self.d ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        out = torch.bmm(attn, vm)  # (B*H, W, d)

        out = out.reshape(b, h, w, self.d).permute(0, 3, 1, 2).contiguous()
        return self.proj(out)

    def forward(self, left, right):
        left_out = self._attend_one(left, right)
        right_out = self._attend_one(right, left)
        return left_out, right_out


class MultiScaleLandmarkEpipolarAttn(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down2 = nn.AvgPool2d(2)
        self.down4 = nn.AvgPool2d(4)

        self.attn_full = LandmarkEpipolarAttn(channels, d=20, num_landmarks=32)
        self.attn_half = LandmarkEpipolarAttn(channels, d=14, num_landmarks=24)
        self.attn_quarter = LandmarkEpipolarAttn(channels, d=10, num_landmarks=16)

    def forward(self, left, right):
        lf, rf = self.attn_full(left, right)

        lh = self.down2(left)
        rh = self.down2(right)
        lh, rh = self.attn_half(lh, rh)
        lh = safe_bilinear_interpolate(lh, scale_factor=2)
        rh = safe_bilinear_interpolate(rh, scale_factor=2)

        lq = self.down4(left)
        rq = self.down4(right)
        lq, rq = self.attn_quarter(lq, rq)
        lq = safe_bilinear_interpolate(lq, scale_factor=4)
        rq = safe_bilinear_interpolate(rq, scale_factor=4)

        left_out = 0.5 * lf + 0.3 * lh + 0.2 * lq
        right_out = 0.5 * rf + 0.3 * rh + 0.2 * rq
        return left_out, right_out


class EfficientDualAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = max(8, channels // 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        c = self.channel_attn(x)
        m = x.mean(dim=1, keepdim=True)
        a = x.amax(dim=1, keepdim=True)
        s = self.spatial_attn(torch.cat([m, a], dim=1))
        return x * (1 + 0.3 * c) * (1 + 0.2 * s)


class MultiScaleMatchingAttention(nn.Module):
    def __init__(self, channels=96):
        super().__init__()
        C = channels

        self.deep_enhance = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 1, bias=False),
        )

        self.multiscale_epipolar = MultiScaleLandmarkEpipolarAttn(C)

        self.enhanced_cross_fusion = nn.Sequential(
            nn.Conv2d(C * 2, C, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 1, bias=True),
        )

        self.dual = EfficientDualAttention(C)

        self.alpha_deep = BoundedAlpha(0.12, 0.0, 0.30)
        self.alpha_multiscale = BoundedAlpha(0.18, 0.0, 0.35)
        self.alpha_cross = BoundedAlpha(0.12, 0.0, 0.30)
        self.alpha_dual = BoundedAlpha(0.10, 0.0, 0.25)

    def forward(self, left, right):
        ld = self.deep_enhance(left)
        rd = self.deep_enhance(right)
        left = left + self.alpha_deep() * ld
        right = right + self.alpha_deep() * rd

        lms, rms = self.multiscale_epipolar(left, right)
        left = left + self.alpha_multiscale() * lms
        right = right + self.alpha_multiscale() * rms

        lc = self.enhanced_cross_fusion(torch.cat([left, right], dim=1))
        rc = self.enhanced_cross_fusion(torch.cat([right, left], dim=1))
        left = left + self.alpha_cross() * lc
        right = right + self.alpha_cross() * rc

        left = left + self.alpha_dual() * self.dual(left)
        right = right + self.alpha_dual() * self.dual(right)

        return left, right
