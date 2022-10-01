'''Utilities for running the pipeline'''
import math
import os
from time import time
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch.autocast_mode import autocast
from PIL import Image

from transformers.models.clip.modeling_clip import CLIPModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler

from guidance import Guide
from pipeline import FlexPipeline

MAX_SEED = 2147483647

sd_model = 'CompVis/stable-diffusion-v1-4'
clip_model = 'openai/clip-vit-large-patch14'
output_dir = './outputs'
grid_dir = f'{output_dir}/grids'

os.makedirs(grid_dir, exist_ok=True)


def _i100(f: float) -> int:
    return int(f * 100)


def image_grid(imgs: Sequence[Image.Image]):
    '''Builds an image that is a grid arrangement of all provided images'''
    num = len(imgs)
    # TODO: account for image height and width to compute ideal grid arrangement
    # TODO: support inputting a desired best fit aspect ration
    # We'll default to width first over height in terms of images not size
    cols = math.ceil(num**(1 / 2)) # sqrt
    rows = num // cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=((i % cols) * w, (i // cols) * h))
    return grid


class Runner():
    def __init__(self, local: bool = True, device: str = 'cuda') -> None:
        if local:
            print('Running local mode only, to download models add --dl')
        else:
            print('Connecting to huggingface to check models...')
        assert FlexPipeline.from_pretrained is not None

        clip = CLIPModel.from_pretrained(clip_model,
                                         local_files_only=local,
                                         use_auth_token=not local)
        assert isinstance(clip, CLIPModel)
        sd: Any = StableDiffusionPipeline.from_pretrained(
            sd_model, local_files_only=local, use_auth_token=not local)
        # TODO: Tweak betas?? can we load??
        # self.scheduler = scheduler = DDIMScheduler(
        #     beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')
        self.pipe = FlexPipeline(sd.vae, clip, sd.tokenizer, sd.unet,
                                 sd.scheduler).to(device)
        self.eta = 0
        self.guide = Guide(clip, self.pipe.tokenizer, device=device)
        self.device = device
        self.generator = torch.Generator(device=device)

    def gen(self,
            prompt: Union[str, List[str]] = '',
            init_image: Optional[Image.Image] = None,
            guide_image: Optional[Image.Image] = None,
            init_size: Tuple[int, int] = (512, 512),
            mapping_concepts: str = '',
            guide_image_threshold_mult: float = 0.5,
            guide_image_threshold_floor: float = 0.5,
            guide_image_clustered: float = 0.5,
            guide_image_linear: Tuple = (0.0, 0.5),
            guide_image_max_guidance: float = 0.5,
            guide_image_mode: int = 0,
            guide_image_reuse: bool = True,
            strength: float = 0.6,
            steps: int = 10,
            guidance_scale: float = 8,
            samples: int = 1,
            seed: Optional[int] = None,
            debug: bool = False):

        fp = f'i2i_ds{int(strength * 100)}' if init_image else 't2i'
        if guide_image:
            fp += (f'_itm{_i100(guide_image_threshold_mult)}'
                   f'_itf{_i100(guide_image_threshold_floor)}'
                   f'_ic{_i100(guide_image_clustered)}'
                   f'_il{_i100(guide_image_linear[0])}'
                   f'-{_i100(guide_image_linear[1])}'
                   f'_im{guide_image_mode:d}')
        fp += f'_st{steps}_gs{int(guidance_scale)}'

        if not seed:
            seed = int(torch.randint(0, MAX_SEED, (1,))[0])
            assert seed is not None
        else:
            seed = min(max(seed, 0), MAX_SEED)
            fp += f'_se{seed}'
        self.generator.manual_seed(seed)

        guide_embeds = self.guide.embeds(
            prompt=prompt,
            guide_image=guide_image,
            mapping_concepts=mapping_concepts,
            guide_image_threshold_mult=guide_image_threshold_mult,
            guide_image_threshold_floor=guide_image_threshold_floor,
            guide_image_clustered=guide_image_clustered,
            guide_image_linear=guide_image_linear,
            guide_image_max_guidance=guide_image_max_guidance,
            guide_image_mode=guide_image_mode,
            guide_image_reuse=guide_image_reuse)

        all_images = []
        for _ in range(samples):
            with autocast(self.device):
                stime = time()
                ms_time = int(stime * 1000)
                with torch.no_grad():
                    output = self.pipe(clip_embeddings=guide_embeds,
                                       init_image=init_image,
                                       init_size=init_size,
                                       strength=strength,
                                       num_inference_steps=steps,
                                       guidance_scale=guidance_scale,
                                       generator=self.generator,
                                       eta=self.eta,
                                       debug=debug)
                images = output['sample'] # type: ignore
                self.eta = time() - stime
                for i, img in enumerate(images):
                    img.save(f'{output_dir}/{ms_time:>013d}_{i:>02d}_{fp}.png',
                             format='png')
            all_images.extend(images)

        ms_time = int(time() * 1000)
        grid = image_grid(all_images)
        grid.save(f'{grid_dir}/{ms_time:>013d}_{fp}.png', format='png')
        return all_images, grid
