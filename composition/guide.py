from math import ceil
from typing import Tuple
import torch
from torch.nn.functional import interpolate
from diffusers.models import UNet2DConditionModel

from encode.clip import CLIPEncoder
from pipeline.guide import GuideBase
from composition.embeds import encode_schema
from composition.schema import Schema

MIN_DIM = 64 # 64 * 8 = 512, the number of pixels that SD generates best images


def _upscale(t: torch.Tensor) -> torch.Tensor:
    low_dim = t.shape[-2]
    if t.shape[-1] < low_dim: # Width < Height
        low_dim = t.shape[-1]
    if low_dim >= MIN_DIM:
        return t
    # Scale up evenly to MIN_DIM
    scale_factor = MIN_DIM / low_dim
    shape = (ceil(t.shape[-2] * scale_factor), ceil(t.shape[-1] * scale_factor))
    return _scale(t, shape)


def _scale(t: torch.Tensor, shape: Tuple[int, int]):
    return interpolate(t, size=shape, mode='bicubic',
                       antialias=True) # type: ignore - bad annotation in torch


class CompositeGuide(GuideBase):
    def __init__(self,
                 encoder: CLIPEncoder,
                 unet: UNet2DConditionModel,
                 guidance: float,
                 schema: Schema,
                 steps: int,
                 batch_size: int = 1):
        GuideBase.__init__(self, encoder, unet, guidance, steps)
        self.schema = schema
        self.embeds = encode_schema(schema, encoder)
        self.batch_size = batch_size
        self.classifier_free_guidance = self.guidance > 1.0
        if self.classifier_free_guidance:
            # Combine uncond then clip embeds into one stack
            self.embed_tensor = torch.cat(
                ([self.uncond_embeds] * self.batch_size) + [
                    self.embeds.background_embed,
                    *[e.embed for e in self.embeds.entities]
                ])
        else:
            self.embed_tensor = torch.cat([
                self.embeds.background_embed,
                *[e.embed for e in self.embeds.entities]
            ])

    def _guide_latents(self, latents: torch.Tensor, embeds: torch.Tensor,
                       step: int) -> torch.FloatTensor:
        in_latents = torch.cat([latents] * self.embed_tensor.shape[0])
        # run unet on all uncond and clip embeds
        noise_pred = self.unet(in_latents,
                               step,
                               encoder_hidden_states=self.embed_tensor).sample
        # Combine all in order of declaration
        if self.classifier_free_guidance:
            noise_stack = noise_pred[self.batch_size:] # First batch is uncod
        else:
            noise_stack = noise_pred
        # Combine
        batch_noise = noise_stack[:self.batch_size]
        entity_noise = noise_stack[self.batch_size:]
        for i in range(self.batch_size):
            bi_noise_pred = batch_noise[i:i + 1, :, :, :]
            # Blend entity noise onto each item in batch noise
            for ei, e in enumerate(self.embeds.entities):
                # latents is shape (batch, channel, height, width)
                ow, oh = e.offset_blocks # Offset
                sw, sh = e.size_blocks # Size
                bw, bh = (ow + sw, oh + sh) # Box
                # Select noise shape
                ens = entity_noise[ei:ei + 1, :, oh:bh, ow:bw]
                # Blend
                bgs = bi_noise_pred[:, :, oh:bh, ow:bw]
                bi_noise_pred[:, :, oh:bh, ow:bw] = bgs + e.blend * (ens - bgs)
            # Set back in batch noise
            batch_noise[i:i + 1, :, :, :] = bi_noise_pred

        if self.classifier_free_guidance:
            # Combine the unconditioned guidance with the clip guidance for bg
            noise_pred_uncond = noise_pred[:self.batch_size]
            batch_noise = noise_pred_uncond + self.guidance * (
                batch_noise - noise_pred_uncond)

        return batch_noise

    def noise_pred(self, latents: torch.Tensor, step: int) -> torch.FloatTensor:
        # noise_pred.shape == latents.shape after chunking
        # before noise_pred.shape == latent_model_input.shape
        # 1, 4, 64, 64 after ; 2, 4, 64, 64 before ; if using 512x512
        # TODO: Composition
        # for compositions we blend in the clip guidance here for each
        # individual entity at it's position, on top of the background
        # embedding
        # This means we have to also trim the latent_model_input based
        # on entity dimensions and run it through the unet to get a
        # noise pred per entity:
        # - uncond_pred for whole canvas
        # - background_pred for whole canvas
        # - entity_pred for each entity in entity block space
        # run unet on all
        bg_embeds = self.embeds.background_embed
        # blend styles
        progress = self.steps / step
        style_base = self.embeds.style_blend[0]
        style_d = self.embeds.style_blend[1] - style_base
        style = self.embeds.style_start_embed + (
            self.embeds.style_end_embed
            - self.embeds.style_start_embed) * (style_base +
                                                (progress * style_d))
        # TODO: Handle style blending
        bg_noise_pred = self._guide_latents(latents, bg_embeds, step)

        # # Run and map all entities
        # for e in self.embeds.entities:
        #     # latents is shape (batch, channel, height, width)
        #     ow, oh = e.offset_blocks # Offset
        #     sw, sh = e.size_blocks # Size
        #     bw, bh = (ow + sw, oh + sh) # Box
        #     l = latents[:, :, oh:bh, ow:bw]
        #     # Upscale before guiding latents as SD works at 512x512
        #     ul = _upscale(l)
        #     enp = self._guide_latents(ul, e.embed, step)
        #     # Downscale back and blend
        #     denp = _scale(enp, (sh, sw))
        #     bgs = bg_noise_pred[:, :, oh:bh, ow:bw]
        #     bg_noise_pred[:, :, oh:bh, ow:bw] = bgs + e.blend * (denp - bgs)

        return bg_noise_pred