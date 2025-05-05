# used but changed from: https://github.com/lucidrains/gigagan-pytorch

import torch
from torch import nn
import torch.nn.functional as F

from src.models.gigagan.utils import (leaky_relu, exists)

# style mapping network

class EqualLinear(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        lr_mul = 1,
        bias = True
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_out))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleNetwork(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        lr_mul = 0.1,
        dim_text_latent = 64
    ):
        super().__init__()
        self.dim = dim
        self.dim_text_latent = dim_text_latent

        layers = []
        for i in range(depth):
            is_first = i == 0
            dim_in = (dim + dim_text_latent) if is_first else dim

            layers.extend([EqualLinear(dim_in, dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x,
        text_latent = None
    ):
        x = F.normalize(x, dim = 1)

        if self.dim_text_latent > 0:
            assert exists(text_latent)
            x = x.to(text_latent.device)
            x = torch.cat((x, text_latent), dim = -1)

        return self.net(x)