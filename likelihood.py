"""Stable Gaussian likelihood of exact, one-sided and interval observations."""

import math
import torch
from torch import nn


def observed_nll(mean, scale, lower, upper):
    if (
        (scale <= 0).any()
        or not torch.isfinite(scale).all()
        or not torch.isfinite(mean).all()
    ):
        raise ValueError("Invalid Gaussian parameters")
    exact = torch.isfinite(lower) & torch.isfinite(upper) & (lower == upper)
    right = torch.isfinite(lower) & torch.isposinf(upper)
    left = torch.isneginf(lower) & torch.isfinite(upper)
    interval = torch.isfinite(lower) & torch.isfinite(upper) & (lower < upper)
    if not (exact | right | left | interval).all():
        raise ValueError("Expected exact, one-sided or ordered finite interval bounds")
    result = torch.empty_like(mean)
    z = (lower[exact] - mean[exact]) / scale[exact]
    result[exact] = 0.5 * z * z + scale[exact].log() + 0.5 * math.log(2 * math.pi)
    result[right] = -torch.special.log_ndtr((mean[right] - lower[right]) / scale[right])
    result[left] = -torch.special.log_ndtr((upper[left] - mean[left]) / scale[left])
    if interval.any():
        # Evaluate in float64, using survival differences in the positive tail.
        a = (lower[interval].double() - mean[interval].double()) / scale[
            interval
        ].double()
        b = (upper[interval].double() - mean[interval].double()) / scale[
            interval
        ].double()
        positive = a > 0
        larger = torch.special.log_ndtr(torch.where(positive, -a, b))
        smaller = torch.special.log_ndtr(torch.where(positive, -b, a))
        delta = smaller - larger
        if (delta >= 0).any():
            raise ValueError(
                "Interval mass is below numeric resolution; widen bounds or rescale observations"
            )
        log_mass = larger + torch.log(-torch.expm1(delta))
        result[interval] = (-log_mass).to(result.dtype)
    return result


class CensoredHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 2)
        )

    def forward(self, x):
        output = self.net(x)
        return output[:, 0], nn.functional.softplus(output[:, 1]) + 0.1
