'''Utilities for running the pipeline'''
import math
import os
from time import time
from typing import Any, List, Sequence, Tuple

import torch
from torch.autocast_mode import autocast
from PIL import Image

from transformers.models.clip.modeling_clip import CLIPModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler

from guidance import Guide
from composition.guide import CompositeGuide
from composition.schema import EntitySchema, Schema
from encode.clip import CLIPEncoder
from pipeline.flex import FlexPipeline
from pipeline.guide import GuideBase, SimpleGuide

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
        self.encoder = CLIPEncoder(clip, self.pipe.tokenizer)
        self.guide = Guide(clip, self.pipe.tokenizer, device=device)
        self.device = device
        self.generator = torch.Generator(device=device)

    def _set_seed(self, seed: int | None):
        if not seed:
            seed = int(torch.randint(0, MAX_SEED, (1,))[0])
        else:
            seed = min(max(seed, 0), MAX_SEED)
        self.generator.manual_seed(seed)

    def _run(self, batches: int, guide: GuideBase,
             init_image: Image.Image | None, init_size: Tuple[int, int],
             strength: float, debug: bool,
             fp: str) -> Tuple[List[Image.Image], Image.Image]:
        all_images = []
        for _ in range(batches):
            with autocast(self.device):
                stime = time()
                ms_time = int(stime * 1000)
                with torch.no_grad():
                    output = self.pipe(guide=guide,
                                       init_image=init_image,
                                       init_size=init_size,
                                       strength=strength,
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

    def gen(self,
            prompt: str | List[str] = '',
            init_image: Image.Image | None = None,
            guide: Image.Image | str | None = None,
            init_size: Tuple[int, int] = (512, 512),
            mapping_concepts: str = '',
            guide_threshold_mult: float = 0.5,
            guide_threshold_floor: float = 0.5,
            guide_clustered: float = 0.5,
            guide_linear: Tuple = (0.0, 0.5),
            guide_max_guidance: float = 0.5,
            guide_header_max: float = 0.15,
            guide_mode: int = 0,
            guide_reuse: bool = True,
            strength: float = 0.6,
            steps: int = 10,
            guidance_scale: float = 8,
            samples: int = 1,
            seed: int | None = None,
            debug: bool = False):

        fp = f'i2i_ds{int(strength * 100)}' if init_image else 't2i'
        if guide:
            fp += (f'_itm{_i100(guide_threshold_mult)}'
                   f'_itf{_i100(guide_threshold_floor)}'
                   f'_ic{_i100(guide_clustered)}'
                   f'_il{_i100(guide_linear[0])}'
                   f'-{_i100(guide_linear[1])}'
                   f'_mg{_i100(guide_max_guidance)}'
                   f'_hm{_i100(guide_header_max)}'
                   f'_im{guide_mode:d}')
        fp += f'_st{steps}_gs{int(guidance_scale)}'
        if seed:
            fp += f'_se{seed}'

        self._set_seed(seed)

        guide_embeds = self.guide.embeds(
            prompt=prompt,
            guide=guide,
            mapping_concepts=mapping_concepts,
            guide_threshold_mult=guide_threshold_mult,
            guide_threshold_floor=guide_threshold_floor,
            guide_clustered=guide_clustered,
            guide_linear=guide_linear,
            guide_max_guidance=guide_max_guidance,
            guide_header_max=guide_header_max,
            guide_mode=guide_mode,
            guide_reuse=guide_reuse)
        pipeline_guide = SimpleGuide(self.encoder, self.pipe.unet,
                                     guidance_scale, steps, guide_embeds)
        return self._run(samples, pipeline_guide, init_image, init_size,
                         strength, debug, fp)

    def compose(self,
                bg_prompt: str = '',
                entities_df: List[List[Any]] = [],
                start_style: str = '',
                end_style: str = '',
                style_blend: Tuple[float, float] = (0.0, 1.0),
                init_image: Image.Image | None = None,
                batches: int = 4,
                strength: float = 0.7,
                steps: int = 30,
                guidance_scale: float = 8.0,
                init_size: Tuple[int, int] = (512, 512),
                seed: int | None = None,
                debug: bool = False):

        fp = f'ci2i_ds{int(strength * 100)}' if init_image else 'ct2i'
        fp += f'_st{steps}_gs{int(guidance_scale)}'
        if seed:
            fp += f'_se{seed}'

        self._set_seed(seed)

        def _row_to_ent(row: List[Any]) -> EntitySchema:
            return EntitySchema(str(row[0]), (int(row[1]), int(row[2])),
                                (int(row[3]), int(row[4])), float(row[5]))

        if hasattr(entities_df, '_values'):
            entities_df = entities_df._values # type: ignore
        schema = Schema(bg_prompt, start_style, end_style, style_blend,
                        [_row_to_ent(r) for r in entities_df])
        pipeline_guide = CompositeGuide(self.encoder, self.pipe.unet,
                                        guidance_scale, schema, steps)
        return self._run(batches, pipeline_guide, init_image, init_size,
                         strength, debug, fp)
