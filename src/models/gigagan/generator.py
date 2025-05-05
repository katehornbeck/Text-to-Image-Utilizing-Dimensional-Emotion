# used but changed from: https://github.com/lucidrains/gigagan-pytorch

from torch import nn, Tensor
import torch.nn.functional as F
import torch

from beartype import beartype
from beartype.typing import List, Tuple, Dict, Iterable
from functools import partial
from einops import repeat

from src.models.gigagan.text_encoder import TextEncoder
from src.models.gigagan.style_network import StyleNetwork
from src.models.gigagan.adaptive_conv import AdaptiveConv2DMod
from src.models.gigagan.utils import (exists, is_power_of_two, log2, SqueezeExcite, Noise, leaky_relu, safe_unshift, is_empty)
from src.models.gigagan.sampling import Upsample
from src.models.gigagan.attention import (SelfAttentionBlock, CrossAttentionBlock)

class BaseGenerator(nn.Module):
    pass

class Generator(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        image_size,
        dim_capacity = 2,
        dim_max = 8,
        channels = 3,
        style_network: StyleNetwork | Dict | None = None,
        style_network_dim = None,
        text_encoder: TextEncoder | Dict | None = None,
        dim_latent = 16,
        self_attn_resolutions: Tuple[int, ...] = (4, 2),
        self_attn_dim_head = 2,
        self_attn_heads = 2,
        self_attn_dot_product = True,
        self_attn_ff_mult = 2,
        cross_attn_resolutions: Tuple[int, ...] = (4, 2),
        cross_attn_dim_head = 4,
        cross_attn_heads = 2,
        cross_attn_ff_mult = 2,
        num_conv_kernels = 1,  # the number of adaptive conv kernels
        num_skip_layers_excite = 0,
        unconditional = False,
        pixel_shuffle_upsample = False
    ):
        super().__init__()
        self.channels = channels

        if isinstance(style_network, dict):
            style_network = StyleNetwork(**style_network)

        self.style_network = style_network

        assert exists(style_network) ^ exists(style_network_dim), 'style_network_dim must be given to the generator if StyleNetwork not passed in as style_network'

        if not exists(style_network_dim):
            style_network_dim = style_network.dim

        self.style_network_dim = style_network_dim

        if isinstance(text_encoder, dict):
            text_encoder = TextEncoder(**text_encoder)

        self.text_encoder = text_encoder

        self.unconditional = unconditional

        assert not (unconditional and exists(text_encoder))
        assert not (unconditional and exists(style_network) and style_network.dim_text_latent > 0)
        assert unconditional or (exists(text_encoder) and text_encoder.dim == style_network.dim_text_latent), 'the `dim_text_latent` on your StyleNetwork must be equal to the `dim` set for the TextEncoder'

        assert is_power_of_two(image_size)
        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers

        # generator requires convolutions conditioned by the style vector
        # and also has N convolutional kernels adaptively selected (one of the only novelties of the paper)

        is_adaptive = num_conv_kernels > 1
        dim_kernel_mod = num_conv_kernels if is_adaptive else 0

        style_embed_split_dims = []

        adaptive_conv = partial(AdaptiveConv2DMod, kernel = 3, num_conv_kernels = num_conv_kernels)

        # initial 4x4 block and conv

        self.init_block = nn.Parameter(torch.randn(dim_latent, 4, 4))
        self.init_conv = adaptive_conv(dim_latent, dim_latent)

        style_embed_split_dims.extend([
            dim_latent,
            dim_kernel_mod
        ])

        # main network

        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers

        resolutions = image_size / ((2 ** torch.arange(num_layers).flip(0)))
        resolutions = resolutions.long().tolist()

        dim_layers = (2 ** (torch.arange(num_layers) + 1)) * dim_capacity
        dim_layers.clamp_(max = dim_max)

        dim_layers = torch.flip(dim_layers, (0,))
        dim_layers = F.pad(dim_layers, (1, 0), value = dim_latent)

        dim_layers = dim_layers.tolist()

        dim_pairs = list(zip(dim_layers[:-1], dim_layers[1:]))

        self.num_skip_layers_excite = num_skip_layers_excite

        self.layers = nn.ModuleList([])

        # go through layers and construct all parameters

        for ind, ((dim_in, dim_out), resolution) in enumerate(zip(dim_pairs, resolutions)):
            is_last = (ind + 1) == len(dim_pairs)
            is_first = ind == 0

            should_upsample = not is_first
            should_upsample_rgb = not is_last
            should_skip_layer_excite = num_skip_layers_excite > 0 and (ind + num_skip_layers_excite) < len(dim_pairs)

            has_self_attn = resolution in self_attn_resolutions
            has_cross_attn = resolution in cross_attn_resolutions and not unconditional

            skip_squeeze_excite = None
            if should_skip_layer_excite:
                dim_skip_in, _ = dim_pairs[ind + num_skip_layers_excite]
                skip_squeeze_excite = SqueezeExcite(dim_in, dim_skip_in)

            resnet_block = nn.ModuleList([
                adaptive_conv(dim_in, dim_out),
                Noise(dim_out),
                leaky_relu(),
                adaptive_conv(dim_out, dim_out),
                Noise(dim_out),
                leaky_relu()
            ])

            to_rgb = AdaptiveConv2DMod(dim_out, channels, 1, num_conv_kernels = 1, demod = False)

            self_attn = cross_attn = rgb_upsample = upsample = None

            upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

            upsample = upsample_klass(dim_in) if should_upsample else None
            rgb_upsample = upsample_klass(channels) if should_upsample_rgb else None

            if has_self_attn:
                self_attn = SelfAttentionBlock(
                    dim_out,
                    dim_head = self_attn_dim_head,
                    heads = self_attn_heads,
                    ff_mult = self_attn_ff_mult,
                    dot_product = self_attn_dot_product
            )

            if has_cross_attn:
                cross_attn = CrossAttentionBlock(
                    dim_out,
                    dim_context = text_encoder.dim,
                    dim_head = cross_attn_dim_head,
                    heads = cross_attn_heads,
                    ff_mult = cross_attn_ff_mult,
                )

            style_embed_split_dims.extend([
                dim_in,             # for first conv in resnet block
                dim_kernel_mod,     # first conv kernel selection
                dim_out,            # second conv in resnet block
                dim_kernel_mod,     # second conv kernel selection
                dim_out,            # to RGB conv
                0,                  # RGB conv kernel selection
            ])

            self.layers.append(nn.ModuleList([
                skip_squeeze_excite,
                resnet_block,
                to_rgb,
                self_attn,
                cross_attn,
                upsample,
                rgb_upsample
            ]))

        # determine the projection of the style embedding to convolutional modulation weights (+ adaptive kernel selection weights) for all layers

        self.style_to_conv_modulations = nn.Linear(style_network_dim, sum(style_embed_split_dims))
        self.style_embed_split_dims = style_embed_split_dims

        self.apply(self.init_)
        nn.init.normal_(self.init_block, std = 0.02)

    def init_(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        styles = None,
        noise = None,
        texts: List[str] | None = None,
        text_encodings: Tensor | None = None,
        global_text_tokens = None,
        fine_text_tokens = None,
        text_mask = None,
        batch_size = 1,
        return_all_rgbs = False
    ):
        # take care of text encodings
        # which requires global text tokens to adaptively select the kernels from the main contribution in the paper
        # and fine text tokens to attend to using cross attention

        if not self.unconditional:
            if exists(texts) or exists(text_encodings):
                assert exists(texts) ^ exists(text_encodings), 'either raw texts as List[str] or text_encodings (from clip) as Tensor is passed in, but not both'
                assert exists(self.text_encoder)

                if exists(texts):
                    text_encoder_kwargs = dict(texts = texts)
                elif exists(text_encodings):
                    text_encoder_kwargs = dict(text_encodings = text_encodings)

                global_text_tokens, fine_text_tokens, text_mask = self.text_encoder(**text_encoder_kwargs)
            else:
                assert all([*map(exists, (global_text_tokens, fine_text_tokens, text_mask))]), 'raw text or text embeddings were not passed in for conditional training'
        else:
            assert not any([*map(exists, (texts, global_text_tokens, fine_text_tokens))])

        # determine styles

        if not exists(styles):
            assert exists(self.style_network)

            if not exists(noise):
                noise = torch.randn((batch_size, self.style_network_dim), device = self.device)

            styles = self.style_network(noise, global_text_tokens)

        # project styles to conv modulations

        conv_mods = self.style_to_conv_modulations(styles)
        conv_mods = conv_mods.split(self.style_embed_split_dims, dim = -1)
        conv_mods = iter(conv_mods)

        # prepare initial block

        batch_size = styles.shape[0]

        x = repeat(self.init_block, 'c h w -> b c h w', b = batch_size)
        x = self.init_conv(x, mod = next(conv_mods), kernel_mod = next(conv_mods))

        rgb = torch.zeros((batch_size, self.channels, 4, 4), device = self.device, dtype = x.dtype)

        # skip layer squeeze excitations

        excitations = [None] * self.num_skip_layers_excite

        # all the rgb's of each layer of the generator is to be saved for multi-resolution input discrimination

        rgbs = []

        # main network

        for squeeze_excite, (resnet_conv1, noise1, act1, resnet_conv2, noise2, act2), to_rgb_conv, self_attn, cross_attn, upsample, upsample_rgb in self.layers:

            if exists(upsample):
                x = upsample(x)

            if exists(squeeze_excite):
                skip_excite = squeeze_excite(x)
                excitations.append(skip_excite)

            excite = safe_unshift(excitations)
            if exists(excite):
                x = x * excite

            x = resnet_conv1(x, mod = next(conv_mods), kernel_mod = next(conv_mods))
            x = noise1(x)
            x = act1(x)

            x = resnet_conv2(x, mod = next(conv_mods), kernel_mod = next(conv_mods))
            x = noise2(x)
            x = act2(x)

            if exists(self_attn):
                x = self_attn(x)

            if exists(cross_attn):
                x = cross_attn(x, context = fine_text_tokens, mask = text_mask)

            layer_rgb = to_rgb_conv(x, mod = next(conv_mods), kernel_mod = next(conv_mods))

            rgb = rgb + layer_rgb

            rgbs.append(rgb)

            if exists(upsample_rgb):
                rgb = upsample_rgb(rgb)

        # sanity check

        assert is_empty([*conv_mods]), 'convolutions were incorrectly modulated'

        if return_all_rgbs:
            return rgb, rgbs

        return rgb