# used but changed from: https://github.com/lucidrains/gigagan-pytorch

from beartype import beartype
from beartype.typing import List, Tuple, Dict, Iterable

from math import log2, sqrt
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.cuda.amp import GradScaler
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange

from src.models.gigagan.distributed import all_gather
from src.models.gigagan.open_clip import OpenClipAdapter

# helpers

def exists(val):
    return val is not None

@beartype
def is_empty(arr: Iterable):
    return len(arr) == 0

def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def is_power_of_two(n):
    return log2(n).is_integer()

def safe_unshift(arr):
    if len(arr) == 0:
        return None
    return arr.pop(0)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def group_by_num_consecutive(arr, num):
    out = []
    for ind, el in enumerate(arr):
        if ind > 0 and divisible_by(ind, num):
            yield out
            out = []

        out.append(el)

    if len(out) > 0:
        yield out

def is_unique(arr):
    return len(set(arr)) == len(arr)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups, remainder = divmod(num, divisor)
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def mkdir_if_not_exists(path):
    path.mkdir(exist_ok = True, parents = True)

@beartype
def set_requires_grad_(
    m: nn.Module,
    requires_grad: bool
):
    for p in m.parameters():
        p.requires_grad = requires_grad

# activation functions

def leaky_relu(neg_slope = 0.2):
    return nn.LeakyReLU(neg_slope)

def conv2d_3x3(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 3, padding = 1)

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gradient_penalty(
    images,
    outputs,
    grad_output_weights = None,
    weight = 10,
    center = 0.,
    scaler: GradScaler | None = None,
    eps = 1e-4
):
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    if exists(scaler):
        outputs = [*map(scaler.scale, outputs)]

    if not exists(grad_output_weights):
        grad_output_weights = (1,) * len(outputs)

    maybe_scaled_gradients, *_ = torch_grad(
        outputs = outputs,
        inputs = images,
        grad_outputs = [(torch.ones_like(output) * weight) for output, weight in zip(outputs, grad_output_weights)],
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )

    gradients = maybe_scaled_gradients

    if exists(scaler):
        scale = scaler.get_scale()
        inv_scale = 1. / max(scale, eps)
        gradients = maybe_scaled_gradients * inv_scale

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - center) ** 2).mean()

# noise

class Noise(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(
        self,
        x,
        noise = None
    ):
        b, _, h, w, device = *x.shape, x.device

        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device = device)

        return x + self.weight * noise

        # hinge gan losses

def generator_hinge_loss(fake):
    return fake.mean()

def discriminator_hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

# auxiliary losses

def aux_matching_loss(real, fake):
    """
    making logits negative, as in this framework, discriminator is 0 for real, high value for fake. GANs can have this arbitrarily swapped, as it only matters if the generator and discriminator are opposites
    """
    return (log(1 + (-real).exp()) + log(1 + (-fake).exp())).mean()

@beartype
def aux_clip_loss(
    clip: OpenClipAdapter,
    images: Tensor,
    texts: List[str] | None = None,
    text_embeds: Tensor | None = None
):
    assert exists(texts) ^ exists(text_embeds)

    images, batch_sizes = all_gather(images, 0, None)

    if exists(texts):
        text_embeds, _ = clip.embed_texts(texts)
        text_embeds, _ = all_gather(text_embeds, 0, batch_sizes)

    return clip.contrastive_loss(images = images, text_embeds = text_embeds)

# differentiable augmentation - Karras et al. stylegan-ada
# start with horizontal flip

class DiffAugment(nn.Module):
    def __init__(
        self,
        *,
        prob,
        horizontal_flip,
        horizontal_flip_prob = 0.5
    ):
        super().__init__()
        self.prob = prob
        assert 0 <= prob <= 1.

        self.horizontal_flip = horizontal_flip
        self.horizontal_flip_prob = horizontal_flip_prob

    def forward(
        self,
        images,
        rgbs: List[Tensor]
    ):
        if random() >= self.prob:
            return images, rgbs

        if random() < self.horizontal_flip_prob:
            images = torch.flip(images, (-1,))
            rgbs = [torch.flip(rgb, (-1,)) for rgb in rgbs]

        return images, rgbs



# skip layer excitation

def SqueezeExcite(dim, dim_out, reduction = 4, dim_min = 32):
    dim_hidden = max(dim_out // reduction, dim_min)

    return nn.Sequential(
        Reduce('b c h w -> b c', 'mean'),
        nn.Linear(dim, dim_hidden),
        nn.SiLU(),
        nn.Linear(dim_hidden, dim_out),
        nn.Sigmoid(),
        Rearrange('b c -> b c 1 1')
    )