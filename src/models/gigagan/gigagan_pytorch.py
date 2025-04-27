from __future__ import annotations

from collections import namedtuple
from beartype import beartype
from beartype.typing import List, Tuple, Dict, Iterable
from pathlib import Path
from math import sqrt

import torch
from torch import nn
from torch.utils.data import DataLoader
from ema_pytorch import EMA
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator
from numerize import numerize
from tqdm import tqdm
from torchvision import utils

from src.models.gigagan.generator import (BaseGenerator, Generator)
from src.models.gigagan.discriminator import (Discriminator, VisionAidedDiscriminator)
from src.models.gigagan.utils import (DiffAugment, exists, mkdir_if_not_exists, cycle, divisible_by, discriminator_hinge_loss, group_by_num_consecutive, aux_matching_loss, generator_hinge_loss, aux_clip_loss, num_to_groups, gradient_penalty)
from src.models.gigagan.optimizer import get_optimizer
from src.models.gigagan.version import __version__

TrainDiscrLosses = namedtuple('TrainDiscrLosses', [
    'divergence',
    'multiscale_divergence',
    'vision_aided_divergence',
    'total_matching_aware_loss',
    'gradient_penalty',
    'aux_reconstruction'
])

TrainGenLosses = namedtuple('TrainGenLosses', [
    'divergence',
    'multiscale_divergence',
    'total_vd_divergence',
    'contrastive_loss'
])

class GigaGAN(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        generator: BaseGenerator | Dict,
        discriminator: Discriminator | Dict,
        vision_aided_discriminator: VisionAidedDiscriminator | Dict | None = None,
        diff_augment: DiffAugment | Dict | None = None,
        learning_rate = 2e-4,
        betas = (0.5, 0.9),
        weight_decay = 0.,
        discr_aux_recon_loss_weight = 1.,
        multiscale_divergence_loss_weight = 0.1,
        vision_aided_divergence_loss_weight = 0.5,
        generator_contrastive_loss_weight = 0.1,
        matching_awareness_loss_weight = 0.1,
        calc_multiscale_loss_every = 1,
        apply_gradient_penalty_every = 4,
        resize_image_mode = 'bilinear',
        train_upsampler = False,
        log_steps_every = 20,
        create_ema_generator_at_init = True,
        save_and_sample_every = 1000,
        early_save_thres_steps = 2500,
        early_save_and_sample_every = 100,
        num_samples = 25,
        model_folder = './data/private/gigagan-models',
        results_folder = './data/private/gigagan-results',
        sample_upsampler_dl: DataLoader | None = None,
        accelerator: Accelerator | None = None,
        accelerate_kwargs: dict = {},
        find_unused_parameters = True,
        amp = True,
        mixed_precision_type = 'fp16'
    ):
        super().__init__()

        # create accelerator

        if accelerator:
            self.accelerator = accelerator
            assert is_empty(accelerate_kwargs)
        else:
            kwargs = DistributedDataParallelKwargs(find_unused_parameters = find_unused_parameters)

            self.accelerator = Accelerator(
                device_placement = False,
                cpu = True,
                kwargs_handlers = [kwargs],
                mixed_precision = mixed_precision_type if amp else 'no',
                split_batches = True,
                **accelerate_kwargs
            )

        # whether to train upsampler or not

        self.train_upsampler = train_upsampler

        if train_upsampler:
            from src.models.gigagan.unet_upsampler import UnetUpsampler
            generator_klass = UnetUpsampler
        else:
            generator_klass = Generator

        # gradient penalty and auxiliary recon loss

        self.apply_gradient_penalty_every = apply_gradient_penalty_every
        self.calc_multiscale_loss_every = calc_multiscale_loss_every

        if isinstance(generator, dict):
            generator = generator_klass(**generator)

        if isinstance(discriminator, dict):
            discriminator = Discriminator(**discriminator)

        if exists(vision_aided_discriminator) and isinstance(vision_aided_discriminator, dict):
            vision_aided_discriminator = VisionAidedDiscriminator(**vision_aided_discriminator)

        assert isinstance(generator, generator_klass)

        # diff augment

        if isinstance(diff_augment, dict):
            diff_augment = DiffAugment(**diff_augment)

        self.diff_augment = diff_augment

        # use _base to designate unwrapped models

        self.G = generator
        self.D = discriminator
        self.VD = vision_aided_discriminator

        # validate multiscale input resolutions

        if train_upsampler:
            assert is_empty(set(discriminator.multiscale_input_resolutions) - set(generator.allowable_rgb_resolutions)), f'only multiscale input resolutions of {generator.allowable_rgb_resolutions} is allowed based on the unet input and output image size. simply do Discriminator(multiscale_input_resolutions = unet.allowable_rgb_resolutions) to resolve this error'

        # ema

        self.has_ema_generator = False

        if self.is_main and create_ema_generator_at_init:
            self.create_ema_generator()

        # print number of parameters

        self.print('\n')

        self.print(f'Generator: {numerize.numerize(generator.total_params)}')
        self.print(f'Discriminator: {numerize.numerize(discriminator.total_params)}')

        if exists(self.VD):
            self.print(f'Vision Discriminator: {numerize.numerize(vision_aided_discriminator.total_params)}')

        self.print('\n')

        # text encoder

        assert generator.unconditional == discriminator.unconditional
        assert not exists(vision_aided_discriminator) or vision_aided_discriminator.unconditional == generator.unconditional

        self.unconditional = generator.unconditional

        # optimizers

        self.G_opt = get_optimizer(self.G.parameters(),  lr = learning_rate, betas = betas, weight_decay = weight_decay)
        self.D_opt = get_optimizer(self.D.parameters(), lr = learning_rate, betas = betas, weight_decay = weight_decay)

        # prepare for distributed

        #self.G = torch.compile(self.G)
        #self.G_opt = torch.compile(self.G_opt)

       # self.G, self.G_opt, _, _ = deepspeed.initialize(model=self.G, config="ds_config.json")
       # self.D, self.D_opt, _, _ = deepspeed.initialize(model=self.D, config="ds_config.json")

        self.G, self.D, self.G_opt, self.D_opt = self.accelerator.prepare(self.G, self.D, self.G_opt, self.D_opt)

        

        # vision aided discriminator optimizer

        if exists(self.VD):
            self.VD_opt = get_optimizer(self.VD.parameters(), lr = learning_rate, betas = betas, weight_decay = weight_decay)
            self.VD_opt = self.accelerator.prepare(self.VD_opt)

        # loss related

        self.discr_aux_recon_loss_weight = discr_aux_recon_loss_weight
        self.multiscale_divergence_loss_weight = multiscale_divergence_loss_weight
        self.vision_aided_divergence_loss_weight = vision_aided_divergence_loss_weight
        self.generator_contrastive_loss_weight = generator_contrastive_loss_weight
        self.matching_awareness_loss_weight = matching_awareness_loss_weight

        # resize image mode

        self.resize_image_mode = resize_image_mode

        # steps

        self.log_steps_every = log_steps_every

        self.register_buffer('steps', torch.ones(1, dtype = torch.long))

        # save and sample

        self.save_and_sample_every = save_and_sample_every
        self.early_save_thres_steps = early_save_thres_steps
        self.early_save_and_sample_every = early_save_and_sample_every

        self.num_samples = num_samples

        self.train_dl = None

        self.sample_upsampler_dl_iter = None
        if exists(sample_upsampler_dl):
            self.sample_upsampler_dl_iter = cycle(self.sample_upsampler_dl)

        self.results_folder = Path(results_folder)
        self.model_folder = Path(model_folder)

        mkdir_if_not_exists(self.results_folder)
        mkdir_if_not_exists(self.model_folder)

    def save(self, path, overwrite = True):
        path = Path(path)
        mkdir_if_not_exists(path.parents[0])

        assert overwrite or not path.exists()

        pkg = dict(
            G = self.unwrapped_G.state_dict(),
            D = self.unwrapped_D.state_dict(),
            G_opt = self.G_opt.state_dict(),
            D_opt = self.D_opt.state_dict(),
            steps = self.steps.item(),
            version = __version__
        )

        if exists(self.G_opt.scaler):
            pkg['G_scaler'] = self.G_opt.scaler.state_dict()

        if exists(self.D_opt.scaler):
            pkg['D_scaler'] = self.D_opt.scaler.state_dict()

        if exists(self.VD):
            pkg['VD'] = self.unwrapped_VD.state_dict()
            pkg['VD_opt'] = self.VD_opt.state_dict()

            if exists(self.VD_opt.scaler):
                pkg['VD_scaler'] = self.VD_opt.scaler.state_dict()

        if self.has_ema_generator:
            pkg['G_ema'] = self.G_ema.state_dict()

        torch.save(pkg, str(path))

    def load(self, path, strict = False):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))

        if 'version' in pkg and pkg['version'] != __version__:
            print(f"trying to load from version {pkg['version']}")

        self.unwrapped_G.load_state_dict(pkg['G'], strict = strict)
        self.unwrapped_D.load_state_dict(pkg['D'], strict = strict)

        if exists(self.VD):
            self.unwrapped_VD.load_state_dict(pkg['VD'], strict = strict)

        if self.has_ema_generator:
            self.G_ema.load_state_dict(pkg['G_ema'])

        if 'steps' in pkg:
            self.steps.copy_(torch.tensor([pkg['steps']]))

        if 'G_opt'not in pkg or 'D_opt' not in pkg:
            return

        try:
            self.G_opt.load_state_dict(pkg['G_opt'])
            self.D_opt.load_state_dict(pkg['D_opt'])

            if exists(self.VD):
                self.VD_opt.load_state_dict(pkg['VD_opt'])

            if 'G_scaler' in pkg and exists(self.G_opt.scaler):
                self.G_opt.scaler.load_state_dict(pkg['G_scaler'])

            if 'D_scaler' in pkg and exists(self.D_opt.scaler):
                self.D_opt.scaler.load_state_dict(pkg['D_scaler'])

            if 'VD_scaler' in pkg and exists(self.VD_opt.scaler):
                self.VD_opt.scaler.load_state_dict(pkg['VD_scaler'])

        except Exception as e:
            self.print(f'unable to load optimizers {e.msg}- optimizer states will be reset')
            pass

    # accelerate related

    @property
    def device(self):
        return self.accelerator.device

    @property
    def unwrapped_G(self):
        return self.accelerator.unwrap_model(self.G)

    @property
    def unwrapped_D(self):
        return self.accelerator.unwrap_model(self.D)

    @property
    def unwrapped_VD(self):
        return self.accelerator.unwrap_model(self.VD)

    @property
    def need_vision_aided_discriminator(self):
        return exists(self.VD) and self.vision_aided_divergence_loss_weight > 0.

    @property
    def need_contrastive_loss(self):
        return self.generator_contrastive_loss_weight > 0. and not self.unconditional

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def resize_image_to(self, images, resolution):
        return F.interpolate(images, resolution, mode = self.resize_image_mode)

    @beartype
    def set_dataloader(self, dl: DataLoader):
        assert not exists(self.train_dl), 'training dataloader has already been set'

        self.train_dl = dl
        self.train_dl_batch_size = dl.batch_size

        self.train_dl = self.accelerator.prepare(self.train_dl)

    # generate function

    @torch.inference_mode()
    def generate(self, *args, **kwargs):
        model = self.G_ema if self.has_ema_generator else self.G
        model.eval()
        return model(*args, **kwargs)

    # create EMA generator

    def create_ema_generator(
        self,
        update_every = 10,
        update_after_step = 100,
        decay = 0.995
    ):
        if not self.is_main:
            return

        assert not self.has_ema_generator, 'EMA generator has already been created'

        self.G_ema = EMA(self.unwrapped_G, update_every = update_every, update_after_step = update_after_step, beta = decay)
        self.has_ema_generator = True

    def generate_kwargs(self, dl_iter, batch_size):
        # what to pass into the generator
        # depends on whether training upsampler or not

        maybe_text_kwargs = dict()
        if self.train_upsampler or not self.unconditional:
            assert exists(dl_iter)

            if self.unconditional:
                real_images = next(dl_iter)
            else:
                result = next(dl_iter)
                assert isinstance(result, tuple), 'dataset should return a tuple of two items for text conditioned training, (images: Tensor, texts: List[str])'
                real_images, texts = result

                maybe_text_kwargs['texts'] = texts[:batch_size]

            real_images = real_images.to(self.device)

        # if training upsample generator, need to downsample real images

        if self.train_upsampler:
            size = self.unwrapped_G.input_image_size
            lowres_real_images = F.interpolate(real_images, (size, size))

            G_kwargs = dict(lowres_image = lowres_real_images)
        else:
            assert exists(batch_size)

            G_kwargs = dict(batch_size = batch_size)

        # create noise

        noise = torch.randn(batch_size, self.unwrapped_G.style_network.dim, device = self.device)

        G_kwargs.update(noise = noise)

        return G_kwargs, maybe_text_kwargs
    
    @beartype
    def train_discriminator_step(
        self,
        dl_iter: Iterable,
        grad_accum_every = 1,
        apply_gradient_penalty = True,
        calc_multiscale_loss = True
    ):
        total_divergence = 0.
        total_vision_aided_divergence = 0.

        total_gp_loss = 0.
        total_aux_loss = 0.

        total_multiscale_divergence = 0. if calc_multiscale_loss else None

        has_matching_awareness = not self.unconditional and self.matching_awareness_loss_weight > 0.

        total_matching_aware_loss = 0.

        all_texts = []
        all_fake_images = []
        all_fake_rgbs = []
        all_real_images = []

        self.G.train()
        self.D.train()

        self.D_opt.zero_grad()
        self.G_opt.zero_grad()

        if self.need_vision_aided_discriminator:
            self.VD.train()
            self.VD_opt.zero_grad()

        for _ in range(grad_accum_every):
            
            if self.unconditional:
                real_images = next(dl_iter)
            else:
                result = next(dl_iter)
                assert isinstance(result, tuple), 'dataset should return a tuple of two items for text conditioned training, (images: Tensor, texts: List[str])'
                real_images, texts = result

                all_real_images.append(real_images)
                all_texts.extend(texts)

            # requires grad for real images, for gradient penalty

            with torch.enable_grad():
                #real_images = real_images.to(self.device)
                real_images = torch.tensor(real_images.clone().detach(), device=self.device, requires_grad=True)

            real_images_rgbs = self.unwrapped_D.real_images_to_rgbs(real_images)

            # diff augment real images

            if exists(self.diff_augment):
                real_images, real_images_rgbs = self.diff_augment(real_images, real_images_rgbs)

            # batch size

            batch_size = real_images.shape[0]

            # for discriminator training, fit upsampler and image synthesis logic under same function

            G_kwargs, maybe_text_kwargs = self.generate_kwargs(dl_iter, batch_size)

            # generator

            with torch.no_grad(), self.accelerator.autocast():
                images, rgbs = self.G(
                    **G_kwargs,
                    **maybe_text_kwargs,
                    return_all_rgbs = True
                )

                all_fake_images.append(images)
                all_fake_rgbs.append(rgbs)

                # diff augment

                if exists(self.diff_augment):
                    images, rgbs = self.diff_augment(images, rgbs)

                # detach output of generator, as training discriminator only

                images.detach_()
                images.requires_grad_()

                for rgb in rgbs:
                    rgb.detach_()
                    rgb.requires_grad_()

            # main divergence loss

            with self.accelerator.autocast():

                fake_logits, fake_multiscale_logits, _ = self.D(
                    images,
                    rgbs,
                    **maybe_text_kwargs,
                    return_multiscale_outputs = calc_multiscale_loss,
                    calc_aux_loss = False
                )

                real_logits, real_multiscale_logits, aux_recon_losses = self.D(
                    real_images,
                    real_images_rgbs,
                    **maybe_text_kwargs,
                    return_multiscale_outputs = calc_multiscale_loss,
                    calc_aux_loss = True
                )

                divergence = discriminator_hinge_loss(real_logits, fake_logits)
                total_divergence += (divergence.item() / grad_accum_every)

                # handle multi-scale divergence

                multiscale_divergence = 0.

                if self.multiscale_divergence_loss_weight > 0. and len(fake_multiscale_logits) > 0:

                    for multiscale_fake, multiscale_real in zip(fake_multiscale_logits, real_multiscale_logits):
                        multiscale_loss = discriminator_hinge_loss(multiscale_real, multiscale_fake)
                        multiscale_divergence = multiscale_divergence + multiscale_loss

                    total_multiscale_divergence += (multiscale_divergence.item() / grad_accum_every)

                # figure out gradient penalty if needed

                gp_loss = 0.

                if apply_gradient_penalty:
                    real_gp_loss = gradient_penalty(
                        real_images,
                        outputs = [real_logits, *real_multiscale_logits],
                        grad_output_weights = [1., *(self.multiscale_divergence_loss_weight,) * len(real_multiscale_logits)],
                        scaler = self.D_opt.scaler
                    )

                    fake_gp_loss = gradient_penalty(
                        images,
                        outputs = [fake_logits, *fake_multiscale_logits],
                        grad_output_weights = [1., *(self.multiscale_divergence_loss_weight,) * len(fake_multiscale_logits)],
                        scaler = self.D_opt.scaler
                    )

                    gp_loss = real_gp_loss + fake_gp_loss

                    if not torch.isnan(gp_loss):
                        total_gp_loss += (gp_loss.item() / grad_accum_every)

                # handle vision aided discriminator, if needed

                vd_loss = 0.

                if self.need_vision_aided_discriminator:

                    fake_vision_aided_logits = self.VD(images, **maybe_text_kwargs)
                    real_vision_aided_logits, clip_encodings = self.VD(real_images, return_clip_encodings = True, **maybe_text_kwargs)

                    for fake_logits, real_logits in zip(fake_vision_aided_logits, real_vision_aided_logits):
                        vd_loss = vd_loss + discriminator_hinge_loss(real_logits, fake_logits)

                    total_vision_aided_divergence += (vd_loss.item() / grad_accum_every)

                    # handle gradient penalty for vision aided discriminator

                    if apply_gradient_penalty:

                        vd_gp_loss = gradient_penalty(
                            clip_encodings,
                            outputs = real_vision_aided_logits,
                            grad_output_weights = [self.vision_aided_divergence_loss_weight] * len(real_vision_aided_logits),
                            scaler = self.VD_opt.scaler
                        )

                        if not torch.isnan(vd_gp_loss):
                            gp_loss = gp_loss + vd_gp_loss

                            total_gp_loss += (vd_gp_loss.item() / grad_accum_every)

                # sum up losses

                total_loss = divergence + gp_loss

                if self.multiscale_divergence_loss_weight > 0.:
                    total_loss = total_loss + multiscale_divergence * self.multiscale_divergence_loss_weight

                if self.vision_aided_divergence_loss_weight > 0.:
                    total_loss = total_loss + vd_loss * self.vision_aided_divergence_loss_weight

                if self.discr_aux_recon_loss_weight > 0.:
                    aux_loss = sum(aux_recon_losses)

                    total_aux_loss += (aux_loss.item() / grad_accum_every)

                    total_loss = total_loss + aux_loss * self.discr_aux_recon_loss_weight

            # backwards

            self.accelerator.backward(total_loss / grad_accum_every)


        # matching awareness loss
        # strategy would be to rotate the texts by one and assume batch is shuffled enough for mismatched conditions

        if has_matching_awareness:

            # rotate texts

            all_texts = [*all_texts[1:], all_texts[0]]
            all_texts = group_by_num_consecutive(texts, batch_size)

            zipped_data = zip(
                all_fake_images,
                all_fake_rgbs,
                all_real_images,
                all_texts
            )

            total_loss = 0.

            for fake_images, fake_rgbs, real_images, texts in zipped_data:

                with self.accelerator.autocast():
                    fake_logits, *_ = self.D(
                        fake_images,
                        fake_rgbs,
                        texts = texts,
                        return_multiscale_outputs = False,
                        calc_aux_loss = False
                    )

                    real_images_rgbs = self.D.real_images_to_rgbs(real_images)

                    real_logits, *_ = self.D(
                        real_images,
                        real_images_rgbs,
                        texts = texts,
                        return_multiscale_outputs = False,
                        calc_aux_loss = False
                    )

                    matching_loss = aux_matching_loss(real_logits, fake_logits)

                    total_matching_aware_loss = (matching_loss.item() / grad_accum_every)

                    loss = matching_loss * self.matching_awareness_loss_weight

                self.accelerator.backward(loss / grad_accum_every)
        torch.cuda.empty_cache()
        self.D_opt.step()
        torch.cuda.empty_cache()

        if self.need_vision_aided_discriminator:
            self.VD_opt.step()

        return TrainDiscrLosses(
            total_divergence,
            total_multiscale_divergence,
            total_vision_aided_divergence,
            total_matching_aware_loss,
            total_gp_loss,
            total_aux_loss
        )

    def train_generator_step(
        self,
        batch_size = None,
        dl_iter: Iterable | None = None,
        grad_accum_every = 1,
        calc_multiscale_loss = True
    ):
        total_divergence = 0.
        total_multiscale_divergence = 0. if calc_multiscale_loss else None
        total_vd_divergence = 0.
        contrastive_loss = 0.

        self.G.train()
        self.D.train()

        self.D_opt.zero_grad()
        self.G_opt.zero_grad()

        all_images = []
        all_texts = []

        torch.cuda.empty_cache()

        for _ in range(grad_accum_every):

            # generator
            
            G_kwargs, maybe_text_kwargs = self.generate_kwargs(dl_iter, batch_size)

            with self.accelerator.autocast():
                images, rgbs = self.G(
                    **G_kwargs,
                    **maybe_text_kwargs,
                    return_all_rgbs = True
                )

                # diff augment

                if exists(self.diff_augment):
                    images, rgbs = self.diff_augment(images, rgbs)

                # accumulate all images and texts for maybe contrastive loss

                if self.need_contrastive_loss:
                    all_images.append(images)
                    all_texts.extend(maybe_text_kwargs['texts'])

                # discriminator

                logits, multiscale_logits, _ = self.D(
                    images,
                    rgbs,
                    **maybe_text_kwargs,
                    return_multiscale_outputs = calc_multiscale_loss,
                    calc_aux_loss = False
                )

                # generator hinge loss discriminator and multiscale

                divergence = generator_hinge_loss(logits)

                total_divergence += (divergence.item() / grad_accum_every)

                total_loss = divergence

                if self.multiscale_divergence_loss_weight > 0. and len(multiscale_logits) > 0:
                    multiscale_divergence = 0.

                    for multiscale_logit in multiscale_logits:
                        multiscale_divergence = multiscale_divergence + generator_hinge_loss(multiscale_logit)

                    total_multiscale_divergence += (multiscale_divergence.item() / grad_accum_every)

                    total_loss = total_loss + multiscale_divergence * self.multiscale_divergence_loss_weight

                # vision aided generator hinge loss

                if self.need_vision_aided_discriminator:
                    vd_loss = 0.

                    logits = self.VD(images, **maybe_text_kwargs)

                    for logit in logits:
                        vd_loss = vd_loss + generator_hinge_loss(logit)

                    total_vd_divergence += (vd_loss.item() / grad_accum_every)

                    total_loss = total_loss + vd_loss * self.vision_aided_divergence_loss_weight

            self.accelerator.backward(total_loss / grad_accum_every, retain_graph = self.need_contrastive_loss)

        # if needs the generator contrastive loss
        # gather up all images and texts and calculate it

        if self.need_contrastive_loss:
            all_images = torch.cat(all_images, dim = 0)

            contrastive_loss = aux_clip_loss(
                clip = self.G.text_encoder.clip,
                texts = all_texts,
                images = all_images
            )

            self.accelerator.backward(contrastive_loss * self.generator_contrastive_loss_weight)

        # generator optimizer step

        torch.cuda.empty_cache()

        #for squeeze_excite, (resnet_conv1, noise1, act1, resnet_conv2, noise2, act2), to_rgb_conv, self_attn, cross_attn, upsample, upsample_rgb in self.G.layers:
        #    torch.cuda.empty_cache()


        self.G_opt.step()
        torch.cuda.empty_cache()
        
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)

        # update exponentially moving averaged generator

        self.accelerator.wait_for_everyone()

        if self.is_main and self.has_ema_generator:
            self.G_ema.update()

        return TrainGenLosses(
            total_divergence,
            total_multiscale_divergence,
            total_vd_divergence,
            contrastive_loss
        )

    def sample(self, model, dl_iter, batch_size):
        G_kwargs, maybe_text_kwargs = self.generate_kwargs(dl_iter, batch_size)

        with self.accelerator.autocast():
            generator_output = model(**G_kwargs, **maybe_text_kwargs)

        if not self.train_upsampler:
            return generator_output

        output_size = generator_output.shape[-1]
        lowres_image = G_kwargs['lowres_image']
        lowres_image = F.interpolate(lowres_image, (output_size, output_size))

        return torch.cat([lowres_image, generator_output])

    @torch.inference_mode()
    def save_sample(
        self,
        batch_size,
        dl_iter = None
    ):
        milestone = self.steps.item() // self.save_and_sample_every
        nrow_mult = 2 if self.train_upsampler else 1
        batches = num_to_groups(self.num_samples, batch_size)

        if self.train_upsampler:
            dl_iter = default(self.sample_upsampler_dl_iter, dl_iter)

        assert exists(dl_iter)

        sample_models_and_output_file_name = [(self.unwrapped_G, f'sample-{milestone}.png')]

        if self.has_ema_generator:
            sample_models_and_output_file_name.append((self.G_ema, f'ema-sample-{milestone}.png'))

        for model, filename in sample_models_and_output_file_name:
            model.eval()

            all_images_list = list(map(lambda n: self.sample(model, dl_iter, n), batches))
            all_images = torch.cat(all_images_list, dim = 0)

            all_images.clamp_(0., 1.)

            utils.save_image(
                all_images,
                str(self.results_folder / filename),
                nrow = int(sqrt(self.num_samples)) * nrow_mult
            )

        # Possible to do: Include some metric to save if improved, include some sampler dict text entries
        self.save(str(self.model_folder / f'model-{milestone}.ckpt'))

    @beartype
    def forward(
        self,
        *,
        steps,
        grad_accum_every = 1
    ):
        torch.cuda.empty_cache()

        assert exists(self.train_dl), 'you need to set the dataloader by running .set_dataloader(dl: Dataloader)'

        batch_size = self.train_dl_batch_size
        dl_iter = cycle(self.train_dl)

        last_gp_loss = 0.
        last_multiscale_d_loss = 0.
        last_multiscale_g_loss = 0.

        for _ in tqdm(range(steps), initial = self.steps.item()):

            torch.cuda.empty_cache()

            steps = self.steps.item()
            is_first_step = steps == 1

            apply_gradient_penalty = self.apply_gradient_penalty_every > 0 and divisible_by(steps, self.apply_gradient_penalty_every)
            calc_multiscale_loss =  self.calc_multiscale_loss_every > 0 and divisible_by(steps, self.calc_multiscale_loss_every)

            (
                d_loss,
                multiscale_d_loss,
                vision_aided_d_loss,
                matching_aware_loss,
                gp_loss,
                recon_loss
            ) = self.train_discriminator_step(
                dl_iter = dl_iter,
                grad_accum_every = grad_accum_every,
                apply_gradient_penalty = apply_gradient_penalty,
                calc_multiscale_loss = calc_multiscale_loss
            )

            self.accelerator.wait_for_everyone()

            (
                g_loss,
                multiscale_g_loss,
                vision_aided_g_loss,
                contrastive_loss
            ) = self.train_generator_step(
                dl_iter = dl_iter,
                batch_size = batch_size,
                grad_accum_every = grad_accum_every,
                calc_multiscale_loss = calc_multiscale_loss
            )

            if exists(gp_loss):
                last_gp_loss = gp_loss

            if exists(multiscale_d_loss):
                last_multiscale_d_loss = multiscale_d_loss

            if exists(multiscale_g_loss):
                last_multiscale_g_loss = multiscale_g_loss

            if is_first_step or divisible_by(steps, self.log_steps_every):

                losses = (
                    ('G', g_loss),
                    ('MSG', last_multiscale_g_loss),
                    ('VG', vision_aided_g_loss),
                    ('D', d_loss),
                    ('MSD', last_multiscale_d_loss),
                    ('VD', vision_aided_d_loss),
                    ('GP', last_gp_loss),
                    ('SSL', recon_loss),
                    ('CL', contrastive_loss),
                    ('MAL', matching_aware_loss)
                )

                losses_str = ' | '.join([f'{loss_name}: {loss:.2f}' for loss_name, loss in losses])

                self.print(losses_str)

            self.accelerator.wait_for_everyone()

            if self.is_main and (is_first_step or divisible_by(steps, self.save_and_sample_every) or (steps <= self.early_save_thres_steps and divisible_by(steps, self.early_save_and_sample_every))):
                self.save_sample(batch_size, dl_iter)
            
            self.steps += 1

        self.print(f'complete {steps} training steps')
