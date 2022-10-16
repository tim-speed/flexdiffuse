from typing import Any, List

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
    '''Preprocess image for encoding

    Args:
        image (Image | Any): PIL Image

    Returns:
        torch.Tensor: Tensor of image data for encoding
    '''
    w, h = image.size
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


class CLIPEncoder():
    def __init__(self, clip: CLIPModel, token: CLIPTokenizer) -> None:
        self.clip = clip
        self.token = token

    def prompt(self, prompt: str | List[str]) -> torch.Tensor:
        '''Encode text embeddings for provided text

        Args:
            text (str | List[str]): Text or array of texts to encode.

        Returns:
            torch.Tensor: Encoded text embeddings.
        '''
        # get prompt text embeddings
        text_input = self.token(
            prompt,
            padding='max_length',
            max_length=self.token.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        return self.clip.text_model(text_input.input_ids.to(
            self.clip.device))[0]

    def image(self, image: Image) -> torch.Tensor:
        '''Encode image embeddings for provided image

        Args:
            image (Image): The PIL.Image to encode.

        Returns:
            torch.Tensor: Encoded image embeddings.
        '''
        guide_tensor = preprocess(image)
        crop_size = min(guide_tensor.shape[-2:])
        guide_tensor = center_crop(guide_tensor, [crop_size, crop_size])
        guide_tensor = resize(guide_tensor, [CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE],
                              interpolation=InterpolationMode.BICUBIC,
                              antialias=True)
        guide_tensor = normalize(
            guide_tensor, [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]).to(self.clip.device)

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

        return self.clip.visual_projection(pooled_output)