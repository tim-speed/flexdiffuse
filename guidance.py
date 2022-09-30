'''Utilities for building prompt or image guided embeddings and tweening the
space of their numbers.'''
from itertools import pairwise
import math
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


def _map_emb(img_emb: torch.Tensor,
             txt_emb: torch.Tensor,
             img_emb_reuse: bool = True,
             guide_order: int = GUIDE_ORDER_ALIGN) -> np.ndarray:
    '''Map the provided image embeddings with the provided text embeddings,\
        according to the provided params to their highest alignment match.

    Args:
        img_emb (torch.Tensor): The image embeddings.
        txt_emb (torch.Tensor): The text embeddings.
        img_emb_reuse (bool, optional): Image embedding reuse, True allows\
            embeddings to be allocated to multiple text embeddings, otherwise\
            they will be used once, based on `guide_order`. Defaults to True.
        guide_order (int, optional): Guide order, prioritize text order or \
            best aligment? Defaults to GUIDE_ORDER_ALIGN.

    Returns:
        np.ndarray: The mappings of shape(MAX_TOKENS-1, 2): (Image_embed_index,\
            Alignment)
    '''
    imgft = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txtft = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    # All token matches is 256 * 77: imf_i, tf_i alignment
    all_matches: List[Tuple[int, int, float]] = []
    # TODO-OPT: Can probably vectorize this
    for i, iimgft in enumerate(imgft[0, :]):
        iimgft = iimgft.unsqueeze(0)
        similarity = (100.0 * (iimgft @ txtft.mT)).softmax(dim=-1)
        # Enumerate matches and similarity, we remove the first index here
        #   so our range goes from 0:MAX_TOKENS -> 1:MAX_TOKENS-1
        #   to ignore the header token.
        all_matches += [
            (i, ii, v.item()) for ii, v in enumerate(similarity[0, 0, 1:])
        ]
    if guide_order == GUIDE_ORDER_TEXT:
        # sort: asc text feature, desc alignment, asc image feature
        all_matches.sort(key=lambda t: (t[1], -t[2], t[0]))
    else:
        # sort: desc alignment, asc text feature, asc image feature
        all_matches.sort(key=lambda t: (-t[2], t[1], t[0]))
    # Now map the img token per text token, without reusing tokens
    #   and skipping the first token, as it is a header not meant to
    #   be changed.
    EDITABLE_TOKENS = txt_emb.shape[1] - 1
    assert EDITABLE_TOKENS == 76, (f'EDITABLE_TOKENS expected to '
                                   f'be 76 got {EDITABLE_TOKENS}')
    mapped_tokens = np.zeros((EDITABLE_TOKENS, 2))
    img_toks_used: Set[int] = set()
    # TODO-OPT: Optimize
    for img_i, txt_i, s in all_matches:
        if mapped_tokens[txt_i, 1] > 0 or img_i in img_toks_used:
            continue
        mapped_tokens[txt_i] = (img_i, s)
        if not img_emb_reuse:
            img_toks_used.add(img_i)
    return mapped_tokens


def _traverse_a_to_b(al: List[int], bl: List[int], weights: torch.Tensor,
                     slope: float) -> torch.Tensor:
    '''Utility for applying linear slope adjustment on weights between points\
        al and bl.

    Args:
        al (List[int]): The points to traverse from.
        bl (List[int]): The points to traverse to.
        weights (torch.Tensor): The weight of each point.
        slope (float): Slope to apply.

    Returns:
        torch.Tensor: The weight tensor modified in place.
    '''
    bi = 0

    def traverse_left(a: int, b: int):
        d = a - b
        gslope = slope / d
        for i in range(1, d):
            weights[a - i] -= gslope * i

    def traverse_right(a: int, b: int):
        d = b - a
        gslope = slope / d
        for i in range(1, d + 1):
            weights[a + i] -= gslope * i

    if bl[0] == 0:
        # Apply full slope on left most point as our algo is right focused
        weights[0] -= slope
    for a in al:
        # Left
        b = bl[bi]
        if b < a:
            traverse_left(a, b)
            bi += 1
        # Right
        if bi >= len(bl):
            # Peak at end
            break
        b = bl[bi]
        traverse_right(a, b)

    return weights


def _clustered_guidance(mapped_tokens: np.ndarray, threshold: float,
                        guidance: float) -> None | torch.Tensor:
    '''Cluster by indentifying all peaks that are over avg_similarity and then\
        and then traversing downward into the valleys as style.
    Similarity Peaks == Subject, Valleys == Style
    Slope will be calculated between peaks and valleys

    Args:
        mapped_tokens (np.ndarray): The mapped tokens to traverse
        threshold (float): The mapping aligment threshold for potential peaks.
        guidance (float): The amount of guidance to apply ( multiplier ).

    Returns:
        None | torch.Tensor: Clustered embedding weights
    '''
    EDITABLE_TOKENS = mapped_tokens.shape[0]
    assert EDITABLE_TOKENS == 76, (f'EDITABLE_TOKENS expected to '
                                   f'be 76 got {EDITABLE_TOKENS}')
    clustered_weights = None
    peaks: List[int] = []
    for txt_i, (_, s) in enumerate(mapped_tokens[1:-1], 1):
        if s < threshold:
            continue
        if (mapped_tokens[txt_i - 1, 1] <= s >= mapped_tokens[txt_i + 1, 1]):
            peaks.append(txt_i)
    if peaks:
        valleys: List[int] = []
        if peaks[0] != 0:
            valleys.append(0)
        for p1, p2 in pairwise(peaks):
            d = p2 - p1
            if d > 0:
                valleys.append(p1 + math.ceil(d / 2))
        if peaks[-1] != EDITABLE_TOKENS - 1:
            valleys.append(EDITABLE_TOKENS - 1)
        # Peaks to Valleys
        clustered_weights = _traverse_a_to_b(
            peaks, valleys, torch.ones((EDITABLE_TOKENS,)), 1.0) * guidance
    return clustered_weights


def _blend_weights(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''Blend the weights described in two tensors into one.

    Args:
        a (torch.Tensor): First tensor to blend
        b (torch.Tensor): Second tensor to blend

    Returns:
        torch.Tensor: A tensor blended of the two weights
    '''
    assert a.shape == b.shape, f'Tensor shapes a={a.shape} != b={b.shape}'
    if a.max() >= 0:
        if b.max() >= 0:
            return torch.maximum(a, b)
        # Fighting eachother
        # TODO: might be a better way?
        return a + b
    # Both negative or zero
    return torch.minimum(a, b)


def preprocess(image: Image | Any) -> torch.Tensor:
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


class Guide():
    def __init__(self,
                 clip: CLIPModel,
                 tokenizer: CLIPTokenizer,
                 device: str = 'cuda') -> None:
        '''Init context for generating prompt or image embeddings and tweening\
            the space of their numbers.

        Args:
            clip (CLIPModel): The CLIP model to use
            tokenizer (CLIPTokenizer): The tokenizer used by the CLIP model.
            device (str, optional): Only tested with cuda. Defaults to 'cuda'.
        '''
        # TODO: Support regular CLIP not just huggingface
        self.clip = clip
        self.tokenizer = tokenizer
        self.device = device
        # Placeholder embed is used for its header token in direct image
        #   guidance
        placeholder_tokens = self.tokenizer(
            '{}',
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        self.placeholder_embed = self.clip.text_model(
            placeholder_tokens.input_ids.to(self.device))[0]

    def embeds(self,
               prompt: str | List[str] = '',
               guide_image: Image | None = None,
               mapping_concepts: str = '',
               guide_image_threshold_mult: float = 0.5,
               guide_image_threshold_floor: float = 0.5,
               guide_image_clustered: float = 0.5,
               guide_image_linear: float = 0.5,
               guide_image_max_guidance: float = 0.5,
               guide_image_mode: int = GUIDE_ORDER_ALIGN,
               guide_image_reuse: bool = True) -> torch.Tensor:
        '''Generate embeddings from text or image or tween their space of\
            numbers.

        Args:
            prompt (str | List[str], optional): The guidance prompt. Defaults\
                to ''.
            guide_image (Image | None, optional): The guidance image. Defaults\
                to None.
            mapping_concepts (str, optional): Concepts to fully map from image.\
                Defaults to ''.
            guide_image_threshold_mult (float, optional): Multiplier for\
                alignment threshold tweening. Defaults to 0.5.
            guide_image_threshold_floor (float, optional): Floor to accept\
                embeddings for threshold tweening based on their alignment.\
                Defaults to 0.5.
            guide_image_clustered (float, optional): A clustered match guidance\
                approach, not as good as threshold but can make some minor\
                adjustments. Defaults to 0.5.
            guide_image_linear (float, optional): Linear style blending from\
                the end of the prompt, good for mapping subtle or background\
                styles from image to text. Defaults to 0.5.
            guide_image_max_guidance (float, optional): Cap on the overall\
                tweening, regardless of multiplier, does not affect mapping\
                concepts. Defaults to 0.5.
            guide_image_mode (int, optional): Image guidance mode, to priorize\
                based on aligment first or text embedding order, effects most\
                noticiable when playing with the `reuse` parameter. Defaults\
                to GUIDE_ORDER_ALIGN.
            guide_image_reuse (bool, optional): Allow re-mapping already mapped\
                image concepts, True will result in a best fit between image\
                and text, but less "uniqueness". Defaults to True.

        Raises:
            ValueError: If you don't supply a proper prompt or guide image.

        Returns:
            torch.Tensor: CLIP embeddings for use in StableDiffusion.
        '''

        if isinstance(prompt, str):
            prompt = prompt.strip()
        elif isinstance(prompt, list):
            prompt = [ss for ss in (s.strip() for s in prompt) if ss]
        else:
            raise ValueError(f'`prompt` has to be of type `str` '
                             f'or `list` but is {type(prompt)}')

        if not prompt and guide_image is None:
            raise ValueError('No prompt, or guide image provided.')

        # TODO-OPT: Remove / Refactor
        CLIP_MAX_TOKENS = self.tokenizer.model_max_length
        assert CLIP_MAX_TOKENS == 77
        EDITABLE_TOKENS = CLIP_MAX_TOKENS - 1 # The first token is not editable

        text_input = None
        text_embeddings: torch.Tensor | None = None
        image_embeddings: torch.Tensor | None = None
        concept_embeddings: torch.Tensor | None = None
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
            guide_tensor = normalize(
                guide_tensor, [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]).to(self.device)

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

            image_embeddings = self.clip.visual_projection(pooled_output)

        if mapping_concepts and image_embeddings is not None:
            concept_input = self.tokenizer(
                mapping_concepts,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            )
            concept_embeddings = self.clip.text_model(
                concept_input.input_ids.to(self.device))[0]
            assert concept_embeddings is not None
            concept_mappings = _map_emb(image_embeddings, concept_embeddings,
                                        False, GUIDE_ORDER_TEXT)
            # Print the result
            print(f'Image Feature and Concept alignment:')
            for txt_i, (img_i, s) in enumerate(concept_mappings, 1):
                print(f'ConceptTok {txt_i:>02d} ImgTok '
                      f'{int(img_i):>02d} {100 * s:.2f}%')

        def _tween(img_emb: torch.Tensor,
                   txt_emb: torch.Tensor) -> torch.Tensor:
            mapped_tokens = _map_emb(img_emb, txt_emb, guide_image_reuse,
                                     guide_image_mode)
            avg_similarity = mapped_tokens[:, 1].mean()
            print(f'Avg Similarity: {avg_similarity:.2%}, '
                  f'Threshold: {guide_image_threshold_floor:.2%}, '
                  f'Threshold Multiplier: {guide_image_threshold_mult:.2%}, '
                  f'Clustered: {guide_image_clustered:.2%}, '
                  f'Linear: {guide_image_linear:.2%}, '
                  f'Guidance Max: {guide_image_max_guidance:.2%}')
            # TODO: Guidance slope param to make either linear or grouped
            #   slopes quadratic .. AKA nice an smooth instead of sharp
            # Init img weights from linear slope, front to back, to amplify
            #   backend / style features
            img_weights = torch.linspace(
                0.0, 1.0, steps=EDITABLE_TOKENS) * guide_image_linear
            if guide_image_clustered != 0:
                clustered_weights = _clustered_guidance(mapped_tokens,
                                                        avg_similarity,
                                                        guide_image_clustered)
                if clustered_weights is not None:
                    img_weights = _blend_weights(img_weights, clustered_weights)

            if guide_image_threshold_mult != 0:
                th_weights = torch.ones(
                    (EDITABLE_TOKENS,)) * guide_image_threshold_mult
                for txt_i, (_, s) in enumerate(mapped_tokens):
                    if s < guide_image_threshold_floor:
                        th_weights[txt_i] = 0
                img_weights = _blend_weights(img_weights, th_weights)

            img_weights = torch.cat((torch.zeros((1,)), img_weights), dim=0)
            print('Image Weights:', img_weights.shape, ':', img_weights)
            # tween text and image embeddings
            # TODO-OPT: Vectorize, probably don't need if conditions
            clip_embeddings = torch.zeros_like(txt_emb)
            for txt_i, (img_i, s) in enumerate(mapped_tokens):
                sd = 1.0 - s
                iw = min(img_weights[txt_i].item(), guide_image_max_guidance)
                if iw == 0:
                    # Leave as is
                    clip_embeddings[0, txt_i] = txt_emb[0, txt_i]
                elif abs(iw) >= sd:
                    # We cap at taking all the image
                    clip_embeddings[0, txt_i] = img_emb[0, int(img_i)]
                else:
                    # Text towards image
                    d_emb = img_emb[0, int(img_i)] - txt_emb[0, txt_i]
                    clip_embeddings[0, txt_i] = txt_emb[0, txt_i] + (d_emb * iw)

            # TODO: Add more customization for this feature, combine with max
            #   guidance etc..
            if concept_embeddings is not None:
                concept_text = _map_emb(concept_embeddings, txt_emb, True,
                                        GUIDE_ORDER_ALIGN)
                # Print the result
                print(f'Concept Feature and Token alignment:')
                for txt_i, (concept_i, s) in enumerate(concept_text, 1):
                    concept_i = int(concept_i)
                    cmi = int(concept_i) - 1 # because mappings start from 1
                    if cmi < 0:
                        # skip header token, it'd be weird for it to map though
                        continue
                    concept_image_i, concept_image_s = concept_mappings[cmi]
                    concept_image_i = int(concept_image_i)
                    if s > 0.9:
                        clip_embeddings[0, txt_i] = img_emb[0, concept_image_i]
                    print(f'TxtTok {txt_i:>02d} ConceptTok '
                          f'{concept_i:>02d} {s:.2%} ImageTok '
                          f'{concept_image_i:>03d} {concept_image_s:.2%}'
                          + (' MAPPED' if s > 0.9 else ''))
            print('Tweened text and image embeddings:', img_emb.shape,
                  ' text shape:', txt_emb.shape, ' embed shape:',
                  clip_embeddings.shape)
            return clip_embeddings

        if text_embeddings is not None:
            if image_embeddings is not None:
                if text_embeddings.shape[0] > 1:
                    # Batch
                    clip_embeddings = text_embeddings.clone()
                    # TODO-OPT: Vectorize??
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
            print('Warning: trying to guide purely from image, '
                  'this will generate weird stuff, enjoy :)\n'
                  'If you\'re bored try an image of yourself '
                  'and see what the model thinks.')
            clip_embeddings = torch.cat(
                (self.placeholder_embed[:, :1, :],
                 image_embeddings[:, :EDITABLE_TOKENS, :]),
                dim=1)

        return clip_embeddings
