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

from pipeline import FlexPipeline

MAX_SEED = 2147483647

sd_model = 'CompVis/stable-diffusion-v1-4'
clip_model = 'openai/clip-vit-large-patch14'
output_dir = './outputs'
grid_dir = f'{output_dir}/grids'

os.makedirs(grid_dir, exist_ok=True)


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
    def __init__(self, local: bool = True) -> None:
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
                                 sd.scheduler).to('cuda')
        self.eta = 0

    def gen(self,
            prompt: Union[str, List[str]] = '',
            init_image: Optional[Union[torch.Tensor, torch.FloatTensor,
                                       Image.Image]] = None,
            guide_image: Optional[Union[torch.Tensor, torch.FloatTensor,
                                        Image.Image]] = None,
            init_size: Tuple[int, int] = (512, 512),
            prompt_text_vs_image: float = 0.5,
            strength: float = 0.6,
            steps: int = 10,
            guidance_scale: float = 8,
            samples: int = 1,
            seed: Optional[int] = None):

        generator = torch.Generator(device='cuda')
        if not seed:
            seed = int(torch.randint(0, MAX_SEED, (1,))[0])
            assert seed is not None
        generator.manual_seed(seed)

        all_images = []
        for _ in range(samples):
            with autocast('cuda'):
                stime = time()
                ms_time = int(stime * 1000)
                with torch.no_grad():
                    output = self.pipe(
                        prompt=prompt,
                        init_image=init_image,
                        guide_image=guide_image,
                        init_size=init_size,
                        prompt_text_vs_image=prompt_text_vs_image,
                        strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        eta=self.eta)
                images = output['sample'] # type: ignore
                self.eta = time() - stime
                for i, img in enumerate(images):
                    img.save(f'{output_dir}/{ms_time:>013d}_{i:>02d}.png',
                             format='png')
            all_images.extend(images)

        ms_time = int(time() * 1000)
        grid = image_grid(all_images)
        grid.save(f'{grid_dir}/{ms_time:>013d}.png', format='png')
        return all_images, grid
