from typing import List
import torch
from diffusers.models import UNet2DConditionModel

from encode.clip import CLIPEncoder


class GuideBase():
    def __init__(self, encoder: CLIPEncoder, unet: UNet2DConditionModel,
                 guidance: float, steps: int) -> None:
        '''Initialize a guide for guiding noise into images.

        Args:
            encoder (CLIPEncoder): An encoder for encoding text to CLIP latents.
            unet (UNet2DConditionModel): Model for predicting the next\
                stage of noise from provided latents and clip_embeddings.
            guidance (float): Guidance scale as defined in [Classifier-Free\
                Diffusion Guidance](https://arxiv.org/abs/2207.12598).\
                `guidance_scale` is defined as `w` of equation 2. of [Imagen\
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is\
                enabled by setting `guidance_scale > 1`. Higher guidance scale\
                encourages to generate images that are closely linked to the\
                text `prompt`, usually at the expense of lower image quality.
            steps (int): Number of steps to generate the image from noise over.\
                More denoising steps usually lead to a higher quality image at\
                the expense of slower inference.
        '''
        self.encoder = encoder
        self.unet = unet
        self.uncond_embeds = encoder.prompt('')
        self.batch_size = 1
        self.guidance = guidance
        self.steps = steps

    def noise_pred(self, latents: torch.Tensor, step: int) -> torch.FloatTensor:
        raise NotImplementedError('noise_pred must be implemented.')


class SimpleGuide(GuideBase):
    def __init__(self, encoder: CLIPEncoder, unet: UNet2DConditionModel,
                 guidance: float, steps: int, clip_embeds: torch.Tensor):
        GuideBase.__init__(self, encoder, unet, guidance, steps)
        self.embeds = clip_embeds
        self.batch_size = self.embeds.shape[0]

    def noise_pred(self, latents: torch.Tensor, step: int) -> torch.FloatTensor:
        classifier_free_guidance = self.guidance > 1.0
        # run unet on all
        clip_embeddings = self.embeds
        if classifier_free_guidance:
            # Combine uncond then clip embeds into one stack
            clip_embeddings = torch.cat(([self.uncond_embeds] * self.batch_size)
                                        + [clip_embeddings])
            latents = torch.cat([latents] * 2)
        # run unet on all uncond and clip embeds
        noise_pred = self.unet(latents,
                               step,
                               encoder_hidden_states=clip_embeddings).sample
        if classifier_free_guidance:
            # Combine the unconditioned guidance with the clip guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance * (
                noise_pred_text - noise_pred_uncond)
        return noise_pred


class PromptGuide(SimpleGuide):
    def __init__(self, encoder: CLIPEncoder, unet: UNet2DConditionModel,
                 guidance: float, steps: int, prompt: str | List[str]):
        SimpleGuide.__init__(self, encoder, unet, guidance, steps,
                             encoder.prompt(prompt))
        self.prompt = prompt