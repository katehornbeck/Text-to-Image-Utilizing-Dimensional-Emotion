from torch import nn, Tensor
import torch.nn.functional as F
import torch

from einops import repeat, rearrange, reduce

from src.models.gigagan.utils import exists

# adaptive conv
# the main novelty of the paper - they propose to learn a softmax weighted sum of N convolutional kernels, depending on the text embedding

def get_same_padding(size, kernel, dilation, stride):
    return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

class AdaptiveConv2DMod(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel,
        *,
        demod = True,
        stride = 1,
        dilation = 1,
        eps = 1e-8,
        num_conv_kernels = 1 # set this to be greater than 1 for adaptive
    ):
        super().__init__()
        self.eps = eps

        self.dim_out = dim_out

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.adaptive = num_conv_kernels > 1

        self.weights = nn.Parameter(torch.randn((num_conv_kernels, dim_out, dim, kernel, kernel)))

        self.demod = demod

        nn.init.kaiming_normal_(self.weights, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def forward(
        self,
        fmap,
        mod: Tensor,
        kernel_mod: Tensor | None = None
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b, h = fmap.shape[0], fmap.shape[-2]

        # account for feature map that has been expanded by the scale in the first dimension
        # due to multiscale inputs and outputs

        if mod.shape[0] != b:
            mod = repeat(mod, 'b ... -> (s b) ...', s = b // mod.shape[0])

        if exists(kernel_mod):
            kernel_mod_has_el = kernel_mod.numel() > 0

            assert self.adaptive or not kernel_mod_has_el

            if kernel_mod_has_el and kernel_mod.shape[0] != b:
                kernel_mod = repeat(kernel_mod, 'b ... -> (s b) ...', s = b // kernel_mod.shape[0])

        # prepare weights for modulation

        weights = self.weights

        if self.adaptive:
            weights = repeat(weights, '... -> b ...', b = b)

            # determine an adaptive weight and 'select' the kernel to use with softmax

            assert exists(kernel_mod) and kernel_mod.numel() > 0

            kernel_attn = kernel_mod.softmax(dim = -1)
            kernel_attn = rearrange(kernel_attn, 'b n -> b n 1 1 1 1')

            weights = reduce(weights * kernel_attn, 'b n ... -> b ...', 'sum')

        # do the modulation, demodulation, as done in stylegan2

        mod = rearrange(mod, 'b i -> b 1 i 1 1')

        weights = weights * (mod + 1)

        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k1 k2 -> b o 1 1 1', 'sum').clamp(min = self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c h w -> 1 (b c) h w')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        padding = get_same_padding(h, self.kernel, self.dilation, self.stride)
        fmap = F.conv2d(fmap, weights, padding = padding, groups = b)

        return rearrange(fmap, '1 (b o) ... -> b o ...', b = b)

class AdaptiveConv1DMod(nn.Module):
    """ 1d version of adaptive conv, for time dimension in videogigagan """

    def __init__(
        self,
        dim,
        dim_out,
        kernel,
        *,
        demod = True,
        stride = 1,
        dilation = 1,
        eps = 1e-8,
        num_conv_kernels = 1 # set this to be greater than 1 for adaptive
    ):
        super().__init__()
        self.eps = eps

        self.dim_out = dim_out

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.adaptive = num_conv_kernels > 1

        self.weights = nn.Parameter(torch.randn((num_conv_kernels, dim_out, dim, kernel)))

        self.demod = demod

        nn.init.kaiming_normal_(self.weights, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def forward(
        self,
        fmap,
        mod: Tensor,
        kernel_mod: Tensor | None = None
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b, t = fmap.shape[0], fmap.shape[-1]

        # account for feature map that has been expanded by the scale in the first dimension
        # due to multiscale inputs and outputs

        if mod.shape[0] != b:
            mod = repeat(mod, 'b ... -> (s b) ...', s = b // mod.shape[0])

        if exists(kernel_mod):
            kernel_mod_has_el = kernel_mod.numel() > 0

            assert self.adaptive or not kernel_mod_has_el

            if kernel_mod_has_el and kernel_mod.shape[0] != b:
                kernel_mod = repeat(kernel_mod, 'b ... -> (s b) ...', s = b // kernel_mod.shape[0])

        # prepare weights for modulation

        weights = self.weights

        if self.adaptive:
            weights = repeat(weights, '... -> b ...', b = b)

            # determine an adaptive weight and 'select' the kernel to use with softmax

            assert exists(kernel_mod) and kernel_mod.numel() > 0

            kernel_attn = kernel_mod.softmax(dim = -1)
            kernel_attn = rearrange(kernel_attn, 'b n -> b n 1 1 1')

            weights = reduce(weights * kernel_attn, 'b n ... -> b ...', 'sum')

        # do the modulation, demodulation, as done in stylegan2

        mod = rearrange(mod, 'b i -> b 1 i 1')

        weights = weights * (mod + 1)

        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k -> b o 1 1', 'sum').clamp(min = self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c t -> 1 (b c) t')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        padding = get_same_padding(t, self.kernel, self.dilation, self.stride)
        fmap = F.conv1d(fmap, weights, padding = padding, groups = b)

        return rearrange(fmap, '1 (b o) ... -> b o ...', b = b)