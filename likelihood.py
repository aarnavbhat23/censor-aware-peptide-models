"""Stable Gaussian likelihood of exact and one-sided censored observations."""
import math
import torch
from torch import nn


def observed_nll(mean,scale,lower,upper):
    if (scale <= 0).any() or not torch.isfinite(scale).all() or not torch.isfinite(mean).all():
        raise ValueError('Invalid Gaussian parameters')
    exact = torch.isfinite(lower) & torch.isfinite(upper) & (lower == upper)
    right = torch.isfinite(lower) & torch.isposinf(upper)
    left = torch.isneginf(lower) & torch.isfinite(upper)
    if not (exact | right | left).all(): raise ValueError('Expected exact or one-sided bounds')
    result = torch.empty_like(mean)
    z = (lower[exact]-mean[exact])/scale[exact]
    result[exact] = .5*z*z+scale[exact].log()+.5*math.log(2*math.pi)
    result[right] = -torch.special.log_ndtr((mean[right]-lower[right])/scale[right])
    result[left] = -torch.special.log_ndtr((upper[left]-mean[left])/scale[left])
    return result


class CensoredHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,2))

    def forward(self,x):
        output = self.net(x)
        return output[:,0],nn.functional.softplus(output[:,1])+.1
