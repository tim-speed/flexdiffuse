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

GUIDE_ORDER_TEXT = 0
GUIDE_ORDER_ALIGN = 1


def preprocess(image: Image | Any) -> torch.Tensor:
    w, h = image.size
    if w > MAX_SINGLE_DIM and h > MAX_SINGLE_DIM:
        if h > w:
            w = (int(w / (h / MAX_SINGLE_DIM)) // 64) * 64
            h = MAX_SINGLE_DIM
        elif w > h:
            h = (int(h / (w / MAX_SINGLE_DIM)) // 64) * 64
            w = MAX_SINGLE_DIM
        else:
            h = MAX_SINGLE_DIM
            w = MAX_SINGLE_DIM
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
               prompt_text_vs_image: float = 0.5,
               guide_image_style_vs_subject: float = 0.5,
               guide_image_mode: int = GUIDE_ORDER_TEXT) -> torch.Tensor:

        if isinstance(prompt, str):
            prompt = prompt.strip()
        elif isinstance(prompt, list):
            prompt = [ss for ss in (s.strip() for s in prompt) if ss]
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

        def _tween(img_emb: torch.Tensor,
                   txt_emb: torch.Tensor) -> torch.Tensor:
            imgft = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txtft = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            # All token matches is 256 * 77: imf_i, tf_i alignment
            all_matches: List[Tuple[int, int, float]] = []
            # TODO: Can probably vectorize this
            for i, iimgft in enumerate(imgft[0, :]):
                iimgft = iimgft.unsqueeze(0)
                similarity = (100.0 * (iimgft @ txtft.mT)).softmax(dim=-1)
                all_matches += [
                    (i, ii, v.item()) for ii, v in enumerate(similarity[0, 0])
                ]
            if guide_image_mode == GUIDE_ORDER_TEXT:
                # sort: asc text feature, desc alignment, asc image feature
                all_matches.sort(key=lambda t: (t[1], -t[2], t[0]))
            else:
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
            style_guidance = 1.0 - guide_image_style_vs_subject
            subject_guidance = guide_image_style_vs_subject
            print(
                f'Guidance Max: {max_guidance:.2%}, '
                f'Image: {image_guidance:.2%}, Text: {text_guidance:.2%}, '
                f'Style: {style_guidance:.2%}, Subject: {subject_guidance:.2%}')
            # TODO: Threshold for activation? on style vs subject?? OR:
            # TODO: Adjust linear function weights based on where subject vs
            #   style align from mappings similarities.
            # Create linear weights, subject_guidance 1.0 == [1.0 ... 0.0]
            #   style_guidance 1.0 == [0.0 ... 1.0]
            # TODO: Vectorize:
            linear_weights = torch.ones((CLIP_MAX_TOKENS,))
            if guide_image_style_vs_subject > 0.5:
                # Front to back reduction
                slope = subject_guidance / CLIP_MAX_TOKENS
                for i in range(1, CLIP_MAX_TOKENS):
                    linear_weights[i] -= slope * i
            elif guide_image_style_vs_subject < 0.5:
                # Back to front reduction
                slope = style_guidance / CLIP_MAX_TOKENS
                for i in range(2, CLIP_MAX_TOKENS + 1):
                    linear_weights[-i] -= slope * i
            # tween text and image embeddings
            # TODO: Vectorize:
            clip_embeddings = torch.zeros_like(txt_emb)
            for txt_i, (img_i, s) in enumerate(mapped_tokens):
                ig = image_guidance * linear_weights[txt_i]
                tg = 1.0 - ig
                clip_embeddings[0, txt_i] = ((txt_emb[0, txt_i] * tg) +
                                             (img_emb[0, int(img_i)] * ig))
            # clip_embeddings = txt_emb.roll(1, 1)
            # clip_embeddings[0, 0] = img_emb[0]
            print('Tweened text and image embeddings:', img_emb.shape,
                  ' text shape:', txt_emb.shape, ' embed shape:',
                  clip_embeddings.shape)
            return clip_embeddings

        if text_embeddings is not None:
            if image_embeddings is not None:
                if text_embeddings.shape[0] > 1:
                    # Batch
                    clip_embeddings = text_embeddings.clone()
                    # TODO: Vectorize??
                    for i, txt_emb in enumerate(text_embeddings):
                        clip_embeddings[i] = _tween(image_embeddings, txt_emb)
                else:
                    # Solo
                    clip_embeddings = _tween(image_embeddings, text_embeddings)
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
