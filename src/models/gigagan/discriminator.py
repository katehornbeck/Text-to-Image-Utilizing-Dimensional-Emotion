from beartype import beartype
from beartype.typing import List, Tuple, Dict, Iterable
from functools import partial

from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch

from src.models.gigagan.open_clip import OpenClipAdapter
from src.models.gigagan.text_encoder import TextEncoder
from src.models.gigagan.utils import (is_power_of_two, is_unique, log2, conv2d_3x3, leaky_relu, SqueezeExcite, exists, default, is_empty, safe_unshift, divisible_by)
from src.models.gigagan.sampling import (Downsample, Upsample)
from src.models.gigagan.adaptive_conv import AdaptiveConv2DMod
from src.models.gigagan.attention import SelfAttentionBlock

@beartype
class SimpleDecoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dims: Tuple[int, ...],
        patch_dim: int = 1,
        frac_patches: float = 1.,
        dropout: float = 0.5
    ):
        super().__init__()
        assert 0 < frac_patches <= 1.

        self.patch_dim = patch_dim
        self.frac_patches = frac_patches

        self.dropout = nn.Dropout(dropout)

        dims = [dim, *dims]

        layers = [conv2d_3x3(dim, dim)]

        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Sequential(
                Upsample(dim_in),
                conv2d_3x3(dim_in, dim_out),
                leaky_relu()
            ))

        self.net = nn.Sequential(*layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        fmap,
        orig_image
    ):
        fmap = self.dropout(fmap)

        if self.frac_patches < 1.:
            batch, patch_dim = fmap.shape[0], self.patch_dim
            fmap_size, img_size = fmap.shape[-1], orig_image.shape[-1]

            assert divisible_by(fmap_size, patch_dim), f'feature map dimensions are {fmap_size}, but the patch dim was designated to be {patch_dim}'
            assert divisible_by(img_size, patch_dim), f'image size is {img_size} but the patch dim was specified to be {patch_dim}'

            fmap, orig_image = map(lambda t: rearrange(t, 'b c (p1 h) (p2 w) -> b (p1 p2) c h w', p1 = patch_dim, p2 = patch_dim), (fmap, orig_image))

            total_patches = patch_dim ** 2
            num_patches_recon = max(int(self.frac_patches * total_patches), 1)

            batch_arange = torch.arange(batch, device = self.device)[..., None]
            batch_randperm = torch.randn((batch, total_patches)).sort(dim = -1).indices
            patch_indices = batch_randperm[..., :num_patches_recon]

            fmap, orig_image = map(lambda t: t[batch_arange, patch_indices], (fmap, orig_image))
            fmap, orig_image = map(lambda t: rearrange(t, 'b p ... -> (b p) ...'), (fmap, orig_image))

        recon = self.net(fmap)
        return F.mse_loss(recon, orig_image)

class RandomFixedProjection(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        channel_first = True
    ):
        super().__init__()
        weights = torch.randn(dim, dim_out)
        nn.init.kaiming_normal_(weights, mode = 'fan_out', nonlinearity = 'linear')

        self.channel_first = channel_first
        self.register_buffer('fixed_weights', weights)

    def forward(self, x):
        if not self.channel_first:
            return x @ self.fixed_weights

        return einsum('b c ..., c d -> b d ...', x, self.fixed_weights)

class VisionAidedDiscriminator(nn.Module):
    """ the vision-aided gan loss """

    @beartype
    def __init__(
        self,
        *,
        depth = 2,
        dim_head = 64,
        heads = 8,
        clip: OpenClipAdapter | None = None,
        layer_indices = (-1, -2, -3),
        conv_dim = None,
        text_dim = None,
        unconditional = False,
        num_conv_kernels = 2
    ):
        super().__init__()

        if not exists(clip):
            clip = OpenClipAdapter()

        self.clip = clip
        dim = clip._dim_image_latent

        self.unconditional = unconditional
        text_dim = default(text_dim, dim)
        conv_dim = default(conv_dim, dim)

        self.layer_discriminators = nn.ModuleList([])
        self.layer_indices = layer_indices

        conv_klass = partial(AdaptiveConv2DMod, kernel = 3, num_conv_kernels = num_conv_kernels) if not unconditional else conv2d_3x3

        for _ in layer_indices:
            self.layer_discriminators.append(nn.ModuleList([
                RandomFixedProjection(dim, conv_dim),
                conv_klass(conv_dim, conv_dim),
                nn.Linear(text_dim, conv_dim) if not unconditional else None,
                nn.Linear(text_dim, num_conv_kernels) if not unconditional else None,
                nn.Sequential(
                    conv2d_3x3(conv_dim, 1),
                    Rearrange('b 1 ... -> b ...')
                )
            ]))

    def parameters(self):
        return self.layer_discriminators.parameters()

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters()])

    @beartype
    def forward(
        self,
        images,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        return_clip_encodings = False
    ):

        assert self.unconditional or (exists(text_embeds) ^ exists(texts))

        with torch.no_grad():
            if not self.unconditional and exists(texts):
                self.clip.eval()
                text_embeds = self.clip.embed_texts

        _, image_encodings = self.clip.embed_images(images)

        logits = []

        for layer_index, (rand_proj, conv, to_conv_mod, to_conv_kernel_mod, to_logits) in zip(self.layer_indices, self.layer_discriminators):
            image_encoding = image_encodings[layer_index]

            cls_token, rest_tokens = image_encoding[:, :1], image_encoding[:, 1:]
            height_width = int(sqrt(rest_tokens.shape[-2])) # assume square

            img_fmap = rearrange(rest_tokens, 'b (h w) d -> b d h w', h = height_width)

            img_fmap = img_fmap + rearrange(cls_token, 'b 1 d -> b d 1 1 ') # pool the cls token into the rest of the tokens

            img_fmap = rand_proj(img_fmap)

            if self.unconditional:
                img_fmap = conv(img_fmap)
            else:
                assert exists(text_embeds)

                img_fmap = conv(
                    img_fmap,
                    mod = to_conv_mod(text_embeds),
                    kernel_mod = to_conv_kernel_mod(text_embeds)
                )

            layer_logits = to_logits(img_fmap)

            logits.append(layer_logits)

        if not return_clip_encodings:
            return logits

        return logits, image_encodings

class Predictor(nn.Module):
    def __init__(
        self,
        dim,
        depth = 4,
        num_conv_kernels = 2,
        unconditional = False
    ):
        super().__init__()
        self.unconditional = unconditional
        self.residual_fn = nn.Conv2d(dim, dim, 1)
        self.residual_scale = 2 ** -0.5

        self.layers = nn.ModuleList([])

        klass = nn.Conv2d if unconditional else partial(AdaptiveConv2DMod, num_conv_kernels = num_conv_kernels)
        klass_kwargs = dict(padding = 1) if unconditional else dict()

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                klass(dim, dim, 3, **klass_kwargs),
                leaky_relu(),
                klass(dim, dim, 3, **klass_kwargs),
                leaky_relu()
            ]))

        self.to_logits = nn.Conv2d(dim, 1, 1)

    def forward(
        self,
        x,
        mod = None,
        kernel_mod = None
    ):
        residual = self.residual_fn(x)

        kwargs = dict()

        if not self.unconditional:
            kwargs = dict(mod = mod, kernel_mod = kernel_mod)

        for conv1, activation, conv2, activation in self.layers:

            inner_residual = x

            x = conv1(x, **kwargs)
            x = activation(x)
            x = conv2(x, **kwargs)
            x = activation(x)

            x = x + inner_residual
            x = x * self.residual_scale

        x = x + residual
        return self.to_logits(x)

class Discriminator(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim_capacity = 2,
        image_size,
        dim_max = 8,
        channels = 3,
        attn_resolutions: Tuple[int, ...] = (4, 2),
        attn_dim_head = 2,
        attn_heads = 2,
        self_attn_dot_product = False,
        ff_mult = 2,
        text_encoder: TextEncoder | Dict | None = None,
        text_dim = None,
        filter_input_resolutions: bool = True,
        multiscale_input_resolutions: Tuple[int, ...] = (16, 8),
        multiscale_output_skip_stages: int = 1,
        aux_recon_resolutions: Tuple[int, ...] = (8,),
        aux_recon_patch_dims: Tuple[int, ...] = (2,),
        aux_recon_frac_patches: Tuple[float, ...] = (0.25,),
        aux_recon_fmap_dropout: float = 0.5,
        resize_mode = 'bilinear',
        num_conv_kernels = 1,
        num_skip_layers_excite = 0,
        unconditional = False,
        predictor_depth = 1
    ):
        super().__init__()
        self.unconditional = unconditional
        assert not (unconditional and exists(text_encoder))

        assert is_power_of_two(image_size)
        assert all([*map(is_power_of_two, attn_resolutions)])

        if filter_input_resolutions:
            multiscale_input_resolutions = [*filter(lambda t: t < image_size, multiscale_input_resolutions)]

        assert is_unique(multiscale_input_resolutions)
        assert all([*map(is_power_of_two, multiscale_input_resolutions)])
        assert all([*map(lambda t: t < image_size, multiscale_input_resolutions)])

        self.multiscale_input_resolutions = multiscale_input_resolutions

        assert multiscale_output_skip_stages > 0
        multiscale_output_resolutions = [resolution // (2 ** multiscale_output_skip_stages) for resolution in multiscale_input_resolutions]

        assert all([*map(lambda t: t >= 4, multiscale_output_resolutions)])

        assert all([*map(lambda t: t < image_size, multiscale_output_resolutions)])

        if len(multiscale_input_resolutions) > 0 and len(multiscale_output_resolutions) > 0:
            assert max(multiscale_input_resolutions) > max(multiscale_output_resolutions)
            assert min(multiscale_input_resolutions) > min(multiscale_output_resolutions)

        self.multiscale_output_resolutions = multiscale_output_resolutions

        assert all([*map(is_power_of_two, aux_recon_resolutions)])
        assert len(aux_recon_resolutions) == len(aux_recon_patch_dims) == len(aux_recon_frac_patches)

        self.aux_recon_resolutions_to_patches = {resolution: (patch_dim, frac_patches) for resolution, patch_dim, frac_patches in zip(aux_recon_resolutions, aux_recon_patch_dims, aux_recon_frac_patches)}

        self.resize_mode = resize_mode

        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers
        self.image_size = image_size

        resolutions = image_size / ((2 ** torch.arange(num_layers)))
        resolutions = resolutions.long().tolist()

        dim_layers = (2 ** (torch.arange(num_layers) + 1)) * dim_capacity
        dim_layers = F.pad(dim_layers, (1, 0), value = channels)
        dim_layers.clamp_(max = dim_max)

        dim_layers = dim_layers.tolist()
        dim_last = dim_layers[-1]
        dim_pairs = list(zip(dim_layers[:-1], dim_layers[1:]))

        self.num_skip_layers_excite = num_skip_layers_excite

        self.residual_scale = 2 ** -0.5
        self.layers = nn.ModuleList([])

        upsample_dims = []
        predictor_dims = []
        dim_kernel_attn = (num_conv_kernels if num_conv_kernels > 1 else 0)

        for ind, ((dim_in, dim_out), resolution) in enumerate(zip(dim_pairs, resolutions)):
            is_first = ind == 0
            is_last = (ind + 1) == len(dim_pairs)
            should_downsample = not is_last
            should_skip_layer_excite = not is_first and num_skip_layers_excite > 0 and (ind + num_skip_layers_excite) < len(dim_pairs)

            has_attn = resolution in attn_resolutions
            has_multiscale_output = resolution in multiscale_output_resolutions

            has_aux_recon_decoder = resolution in aux_recon_resolutions
            upsample_dims.insert(0, dim_in)

            skip_squeeze_excite = None
            if should_skip_layer_excite:
                dim_skip_in, _ = dim_pairs[ind + num_skip_layers_excite]
                skip_squeeze_excite = SqueezeExcite(dim_in, dim_skip_in)

            # multi-scale rgb input to feature dimension

            from_rgb = nn.Conv2d(channels, dim_in, 7, padding = 3)

            # residual convolution

            residual_conv = nn.Conv2d(dim_in, dim_out, 1, stride = (2 if should_downsample else 1))

            # main resnet block

            resnet_block = nn.Sequential(
                conv2d_3x3(dim_in, dim_out),
                leaky_relu(),
                conv2d_3x3(dim_out, dim_out),
                leaky_relu()
            )

            # multi-scale output

            multiscale_output_predictor = None

            if has_multiscale_output:
                multiscale_output_predictor = Predictor(dim_out, num_conv_kernels = num_conv_kernels, depth = 2, unconditional = unconditional)
                predictor_dims.extend([dim_out, dim_kernel_attn])

            aux_recon_decoder = None

            if has_aux_recon_decoder:
                patch_dim, frac_patches = self.aux_recon_resolutions_to_patches[resolution]

                aux_recon_decoder = SimpleDecoder(
                    dim_out,
                    dims = tuple(upsample_dims),
                    patch_dim = patch_dim,
                    frac_patches = frac_patches,
                    dropout = aux_recon_fmap_dropout
                )

            self.layers.append(nn.ModuleList([
                skip_squeeze_excite,
                from_rgb,
                resnet_block,
                residual_conv,
                SelfAttentionBlock(dim_out, heads = attn_heads, dim_head = attn_dim_head, ff_mult = ff_mult, dot_product = self_attn_dot_product) if has_attn else None,
                multiscale_output_predictor,
                aux_recon_decoder,
                Downsample(dim_out) if should_downsample else None,
            ]))

        self.to_logits = nn.Sequential(
            conv2d_3x3(dim_last, dim_last),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(dim_last * (4 ** 2), 1),
            Rearrange('b 1 -> b')
        )

        # take care of text conditioning in the multiscale predictor branches

        assert unconditional or (exists(text_dim) ^ exists(text_encoder))

        if not unconditional:
            if isinstance(text_encoder, dict):
                text_encoder = TextEncoder(**text_encoder)

            self.text_dim = default(text_dim, text_encoder.dim)

            self.predictor_dims = predictor_dims
            self.text_to_conv_conditioning = nn.Linear(self.text_dim, sum(predictor_dims)) if exists(self.text_dim) else None

        self.text_encoder = text_encoder

        self.apply(self.init_)

    def init_(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def resize_image_to(self, images, resolution):
        return F.interpolate(images, resolution, mode = self.resize_mode)

    def real_images_to_rgbs(self, images):
        return [self.resize_image_to(images, resolution) for resolution in self.multiscale_input_resolutions]

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters()])

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        images,
        rgbs: List[Tensor],                   # multi-resolution inputs (rgbs) from the generator
        texts: List[str] | None = None,
        text_encodings: Tensor | None = None,
        text_embeds = None,
        real_images = None,                   # if this were passed in, the network will automatically append the real to the presumably generated images passed in as the first argument, and generate all intermediate resolutions through resizing and concat appropriately
        return_multiscale_outputs = True,     # can force it not to return multi-scale logits
        calc_aux_loss = True
    ):
        if not self.unconditional:
            assert (exists(texts) ^ exists(text_encodings)) ^ exists(text_embeds), 'either texts as List[str] is passed in, or clip text_encodings as Tensor'

            if exists(texts):
                assert exists(self.text_encoder)
                text_embeds, *_ = self.text_encoder(texts = texts)

            elif exists(text_encodings):
                assert exists(self.text_encoder)
                text_embeds, *_ = self.text_encoder(text_encodings = text_encodings)

            assert exists(text_embeds), 'raw text or text embeddings were not passed into discriminator for conditional training'

            conv_mods = self.text_to_conv_conditioning(text_embeds).split(self.predictor_dims, dim = -1)
            conv_mods = iter(conv_mods)

        else:
            assert not any([*map(exists, (texts, text_embeds))])

        x = images

        image_size = (self.image_size, self.image_size)

        assert x.shape[-2:] == image_size

        batch = x.shape[0]

        # index the rgbs by resolution

        rgbs_index = {t.shape[-1]: t for t in rgbs} if exists(rgbs) else {}

        # assert that the necessary resolutions are there

        assert is_empty(set(self.multiscale_input_resolutions) - set(rgbs_index.keys())), f'rgbs of necessary resolution {self.multiscale_input_resolutions} were not passed in'

        # hold multiscale outputs

        multiscale_outputs = []

        # hold auxiliary recon losses

        aux_recon_losses = []

        # excitations

        excitations = [None] * (self.num_skip_layers_excite + 1) # +1 since first image in pixel space is not excited

        for squeeze_excite, from_rgb, block, residual_fn, attn, predictor, recon_decoder, downsample in self.layers:
            resolution = x.shape[-1]

            if exists(squeeze_excite):
                skip_excite = squeeze_excite(x)
                excitations.append(skip_excite)

            excite = safe_unshift(excitations)

            if exists(excite):
                excite = repeat(excite, 'b ... -> (s b) ...', s = x.shape[0] // excite.shape[0])
                x = x * excite

            batch_prev_stage = x.shape[0]
            has_multiscale_input = resolution in self.multiscale_input_resolutions

            if has_multiscale_input:
                rgb = rgbs_index.get(resolution, None)

                # multi-scale input features

                multi_scale_input_feats = from_rgb(rgb)

                # expand multi-scale input features, as could include extra scales from previous stage

                multi_scale_input_feats = repeat(multi_scale_input_feats, 'b ... -> (s b) ...', s = x.shape[0] // rgb.shape[0])

                # add the multi-scale input features to the current hidden state from main stem

                x = x + multi_scale_input_feats

                # and also concat for scale invariance

                x = torch.cat((x, multi_scale_input_feats), dim = 0)

            residual = residual_fn(x)
            x = block(x)

            if exists(attn):
                x = attn(x)

            if exists(predictor):
                pred_kwargs = dict()
                if not self.unconditional:
                    pred_kwargs = dict(mod = next(conv_mods), kernel_mod = next(conv_mods))

                if return_multiscale_outputs:
                    predictor_input = x[:batch_prev_stage]
                    multiscale_outputs.append(predictor(predictor_input, **pred_kwargs))

            if exists(downsample):
                x = downsample(x)

            x = x + residual
            x = x * self.residual_scale

            if exists(recon_decoder) and calc_aux_loss:

                recon_output = x[:batch_prev_stage]
                recon_output = rearrange(x, '(s b) ... -> s b ...', b = batch)

                aux_recon_target = images

                # only use the input real images for aux recon

                recon_output = recon_output[0]

                # only reconstruct a fraction of images across batch and scale
                # for efficiency

                aux_recon_loss = recon_decoder(recon_output, aux_recon_target)
                aux_recon_losses.append(aux_recon_loss)

        # sanity check

        assert self.unconditional or is_empty([*conv_mods]), 'convolutions were incorrectly modulated'

        # to logits

        logits = self.to_logits(x)   
        logits = rearrange(logits, '(s b) ... -> s b ...', b = batch)

        return logits, multiscale_outputs, aux_recon_losses