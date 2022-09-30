'''Pipeline, a modification of Img2Img from huggingface/diffusers'''
import inspect
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

import PIL
import PIL.Image

from transformers.models.clip.modeling_clip import CLIPModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (DDIMScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from guidance import preprocess


class FlexPipeline(DiffusionPipeline):
    r'''
    Pipeline for text-guided image to image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        clip ([`CLIPModel`]):
            Frozen CLIPModel. Stable Diffusion traditionally uses the text portion but we'll allow both of:
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    '''
    def __init__(
        self,
        vae: AutoencoderKL,
        clip: CLIPModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
    ):
        super().__init__()
        scheduler = scheduler.set_format('pt')

        if hasattr(scheduler.config,
                   'steps_offset') and scheduler.config['steps_offset'] != 1:
            warnings.warn(
                f'The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`'
                f' should be set to 1 instead of {scheduler.config["steps_offset"]}. Please make sure '
                'to update the config accordingly as leaving `steps_offset` might led to incorrect results'
                ' in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,'
                ' it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`'
                ' file',
                DeprecationWarning,
            )
            new_config = dict(scheduler.config)
            new_config['steps_offset'] = 1
            setattr(scheduler, '_internal_dict', FrozenDict(new_config))

        self.vae = vae
        self.clip = clip
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.register_modules(
            vae=vae,
            clip=clip,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

    def enable_attention_slicing(self,
                                 slice_size: Optional[Union[str,
                                                            int]] = 'auto'):
        r'''
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `'auto'`):
                When `'auto'`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        '''
        if slice_size == 'auto':
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config['attention_head_dim'] // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r'''
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        '''
        # set slice_size = `None` to disable `set_attention_slice`
        self.enable_attention_slicing(None)

    def _latents_to_image(
            self,
            latents: torch.Tensor,
            pil: bool = True) -> Union[np.ndarray, List[PIL.Image.Image]]:
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample # type: ignore

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if pil:
            return self.numpy_to_pil(image)
        return image

    @torch.no_grad()
    def __call__(self,
                 clip_embeddings: torch.Tensor,
                 init_image: Optional[Union[torch.Tensor, torch.FloatTensor,
                                            PIL.Image.Image]] = None,
                 init_size: Tuple[int, int] = (512, 512),
                 strength: float = 0.6,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 eta: float = 0.0,
                 generator: Optional[torch.Generator] = None,
                 output_type: str = 'pil',
                 return_dict: bool = True,
                 debug: bool = False):
        r'''
        Function invoked when calling the pipeline for generation.
        Args:
            clip_embeddings (`torch.Tensor`):
                The prompt embeddings to guide image generation.
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            init_size (`Tuple[int, int]`):
                height and width for the output image initial latents if init_image not provided 
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `'pil'`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is
            False.
        '''
        if strength < 0 or strength > 1:
            raise ValueError(
                f'The value of strength should in [0.0, 1.0] but is {strength}')

        batch_size = clip_embeddings.shape[0]

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        assert self.scheduler.timesteps is not None

        # Handle VAE latent initialization
        if init_image:
            if isinstance(init_image, PIL.Image.Image):
                init_image = preprocess(init_image)
            # TODO: Confirm correct tensor shape expectation
            width, height = init_image.shape[-2:]
            init_image = init_image.to(self.device)

            # encode the init image into latents and scale the latents
            init_latent_dist = self.vae.encode(init_image # type: ignore
                                              ).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents
            # expand init_latents for batch_size
            init_latents = torch.cat([init_latents] * batch_size)

            # get the original timestep using init_timestep
            offset = self.scheduler.config.get('steps_offset', 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                timesteps = torch.tensor([num_inference_steps - init_timestep]
                                         * batch_size,
                                         dtype=torch.long,
                                         device=self.device)
            else:
                timesteps = self.scheduler.timesteps[-init_timestep]
                timesteps = torch.tensor([timesteps] * batch_size,
                                         dtype=torch.long,
                                         device=self.device)

            # add noise to latents using the timesteps
            noise = torch.randn(init_latents.shape,
                                generator=generator,
                                device=self.device)
            init_latents = self.scheduler.add_noise(
                init_latents, # type: ignore
                noise, # type: ignore
                timesteps) # type: ignore

            # Calc start timestep
            t_start = max(num_inference_steps - init_timestep + offset, 0)
        else:
            height, width = init_size
            channels: int = self.unet.in_channels # type: ignore
            # Random init
            init_latents = torch.randn(
                (batch_size, channels, height // 8, width // 8),
                generator=generator,
                device=self.device,
            )

            # set timesteps
            self.scheduler.set_timesteps(num_inference_steps)

            # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                init_latents: torch.Tensor = (
                    init_latents * self.scheduler.sigmas[0]) # type: ignore

            # No init image so we start from 0
            t_start = 0

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            max_length = clip_embeddings.shape[1]
            # TODO: Remove this assert
            assert max_length == 77
            # TODO: Support negative embeddings?
            uncond_input = self.tokenizer([''] * batch_size,
                                          padding='max_length',
                                          max_length=max_length,
                                          return_tensors='pt')
            uncond_embeddings = self.clip.text_model(
                uncond_input.input_ids.to(self.device))[0]
            clip_embeddings = torch.cat([uncond_embeddings, clip_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        latents = init_latents
        all_latents = None
        if debug:
            all_latents = [init_latents]
        for i, t in enumerate(
                self.progress_bar(self.scheduler.timesteps[t_start:])):
            t_index = t

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents

            # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                t_index = t_start + i
                sigma = self.scheduler.sigmas[t_index] # type: ignore
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1)**0.5)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input,
                                   t,
                                   encoder_hidden_states=clip_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            step = self.scheduler.step(
                noise_pred,
                t_index,
                latents, # type: ignore
                **extra_step_kwargs)
            latents = step.prev_sample # type: ignore
            if all_latents:
                all_latents.append(latents)

        if all_latents:
            image_batches = [
                self._latents_to_image(l, output_type == 'pil')
                for l in all_latents
            ]
            if isinstance(image_batches[0], list):
                batch_images = []
                for ib in image_batches:
                    batch_images.extend(ib)
            else:
                batch_images = np.concatenate(
                    image_batches, # type: ignore
                    axis=0)
        else:
            batch_images = self._latents_to_image(latents, output_type == 'pil')

        if not return_dict:
            return (batch_images, False)

        return StableDiffusionPipelineOutput(
            images=batch_images,
            nsfw_content_detected=[False for _ in batch_images])
