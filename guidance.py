from typing import Any, List, Set, Tuple

import numpy as np
from PIL.Image import Image, LANCZOS
import torch
from torchvision.transforms.functional import (center_crop, normalize, resize,
                                               InterpolationMode)
from transformers.models.clip.modeling_clip import CLIPModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

CLIP_IMAGE_SIZE = 224
MAX_SINGLE_DIM = 512 # for stable diffusion image


def preprocess(image: Image | Any) -> torch.Tensor:
    w, h = image.size
    if w > MAX_SINGLE_DIM and h > MAX_SINGLE_DIM:
        if h > w:
            w = int(w / (h / MAX_SINGLE_DIM))
            h = MAX_SINGLE_DIM
        elif w > h:
            h = int(h / (w / MAX_SINGLE_DIM))
            w = MAX_SINGLE_DIM
        else:
            h = MAX_SINGLE_DIM
            w = MAX_SINGLE_DIM
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=LANCZOS)
    image = image.convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class Guide():
    def __init__(self,
                 clip: CLIPModel,
                 tokenizer: CLIPTokenizer,
                 device: str = 'cuda') -> None:
        self.clip = clip
        self.tokenizer = tokenizer
        self.device = device

    def embeds(self,
               prompt: str | List[str] = '',
               guide_image: Image | None = None,
               prompt_text_vs_image: float = 0.5) -> torch.Tensor:

        if isinstance(prompt, str):
            prompt = prompt.strip()
            batch_size = 1
        elif isinstance(prompt, list):
            prompt = [ss for ss in (s.strip() for s in prompt) if ss]
            batch_size = len(prompt)
        else:
            raise ValueError(
                f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )

        if not prompt and guide_image is None:
            raise ValueError('No prompt, or guide image provided.')

        # TODO: Remove / Refactor
        CLIP_MAX_TOKENS = self.tokenizer.model_max_length
        assert CLIP_MAX_TOKENS == 77

        text_input = None
        text_embeddings: torch.Tensor | None = None
        image_embeddings: torch.Tensor | None = None
        if prompt:
            # get prompt text embeddings
            text_input = self.tokenizer(
                prompt,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            )
            text_embeddings = self.clip.text_model(
                text_input.input_ids.to(self.device))[0]
        if guide_image is not None:
            guide_tensor = preprocess(guide_image)
            crop_size = min(guide_tensor.shape[-2:])
            guide_tensor = center_crop(guide_tensor, [crop_size, crop_size])
            guide_tensor = resize(guide_tensor,
                                  [CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE],
                                  interpolation=InterpolationMode.BICUBIC,
                                  antialias=True)
            guide_tensor = normalize(guide_tensor,
                                     [0.48145466, 0.4578275, 0.40821073],
                                     [0.26862954, 0.26130258, 0.27577711])

            # dbgim = (guide_tensor / 2 + 0.5).clamp(0, 1)
            # dbgim = dbgim.cpu().permute(0, 2, 3, 1).numpy()
            # dbgim = self.numpy_to_pil(dbgim)
            # dbgim[0].save('./guide_tensor.png', format='png')

            guide_tensor = guide_tensor.to(self.device)

            hidden_states = self.clip.vision_model.embeddings(guide_tensor)
            hidden_states = self.clip.vision_model.pre_layrnorm(hidden_states)

            encoder_outputs = self.clip.vision_model.encoder(
                inputs_embeds=hidden_states,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            last_hidden_state = encoder_outputs[0]
            pooled_output = last_hidden_state[:, :, :]
            pooled_output = self.clip.vision_model.post_layernorm(pooled_output)

            # pooled_output = vision_outputs[1] # pooled_output
            image_embeddings = self.clip.visual_projection(pooled_output)

            # image_embeddings = self.clip.get_image_features(
            #     guide_image # type: ignore
            # )

        if text_embeddings is not None:
            if image_embeddings is not None:
                assert text_input is not None
                imgft = image_embeddings / image_embeddings.norm(dim=-1,
                                                                 keepdim=True)
                txtft = text_embeddings / text_embeddings.norm(dim=-1,
                                                               keepdim=True)
                # All token matches is 256 * 77: imf_i, tf_i alignment
                all_matches: List[Tuple[int, int, float]] = []
                # TODO: Can probably vectorize this
                for i, iimgft in enumerate(imgft[0, :]):
                    iimgft = iimgft.unsqueeze(0)
                    similarity = (100.0 * (iimgft @ txtft.mT)).softmax(dim=-1)
                    all_matches += [(i, ii, v.item())
                                    for ii, v in enumerate(similarity[0, 0])]
                # sort: desc alignment, asc text feature, asc image feature
                all_matches.sort(key=lambda t: (-t[2], t[1], t[0]))
                # Now map the img token per text token, without reusing tokens
                mapped_tokens = np.zeros((CLIP_MAX_TOKENS, 2))
                img_toks_used: Set[int] = set()
                # TODO: Optimize
                for img_i, txt_i, s in all_matches:
                    if mapped_tokens[txt_i, 1] > 0 or img_i in img_toks_used:
                        continue
                    mapped_tokens[txt_i] = (img_i, s)
                    img_toks_used.add(img_i)
                # Print the result
                print(f'Image Feature and Token alignment:')
                for txt_i, (img_i, s) in enumerate(mapped_tokens):
                    print(f'TxtTok {txt_i:>02d} ImgTok '
                          f'{int(img_i):>02d} {100 * s:.2f}%')
                # TODO: Max Guidance options
                max_guidance = 1 - mapped_tokens[:, 1].mean()
                image_guidance = max_guidance * prompt_text_vs_image
                text_guidance = 1.0 - image_guidance
                print(f'Guidance Max: {max_guidance:.2%}, Image: '
                      f'{image_guidance:.2%}, Text: {text_guidance:.2%}')
                # TODO: Map options AKA sort.. ( by aligment (Current), by Text Order (additional) )
                # TODO: Threshold for activation?
                # TODO: Linear scale downward for less aligned tokens?
                # tween text and image embeddings
                clip_embeddings = text_embeddings * text_guidance
                for txt_i, (img_i, s) in enumerate(mapped_tokens):
                    clip_embeddings[0, txt_i] += image_embeddings[
                        0, int(img_i)] * image_guidance
                # clip_embeddings = text_embeddings.roll(1, 1)
                # clip_embeddings[0, 0] = image_embeddings[0]
                print('Tweening text and image embeddings:',
                      image_embeddings.shape, ' text shape:',
                      text_embeddings.shape, ' embed shape:',
                      clip_embeddings.shape)
            else:
                clip_embeddings: torch.Tensor = text_embeddings
        else:
            assert image_embeddings is not None
            # Select the first CLIP_MAX_TOKENS image embedding tokens only
            # NOTE: This is not good guidance, StableDiffusion wasn't trained
            #   for this. We need to map to prompts.
            # TODO: Build a model that can resequence image embeddings to
            #   a similar structure as text, SEE: BLIP ?? No need for text tho.
            clip_embeddings: torch.Tensor = image_embeddings[:, :
                                                             CLIP_MAX_TOKENS, :]

        return clip_embeddings
