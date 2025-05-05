# used but changed from: https://github.com/lucidrains/gigagan-pytorch

from torch import nn

from src.models.gigagan.attention import (TextAttention, FeedForward)
from src.models.gigagan.rms import RMSNorm

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TextAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.norm(x)