from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from composition.schema import EntitySchema, Schema
from encode.clip import CLIPEncoder


@dataclass
class EntityEmbeds():
    embed: torch.Tensor
    offset_blocks: Tuple[int, int]
    size_blocks: Tuple[int, int]
    blend: float
    # TODO: Shape / mask


@dataclass
class Embeds():
    background_embed: torch.Tensor
    style_start_embed: torch.Tensor
    style_end_embed: torch.Tensor
    style_blend: Tuple[float, float]
    entities: List[EntityEmbeds]


def px_to_block(px_shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(px // 8 for px in px_shape)


def encode_entity(e: EntitySchema, encode: CLIPEncoder) -> EntityEmbeds:
    return EntityEmbeds(embed=encode.prompt(e.prompt),
                        offset_blocks=px_to_block(e.offset),
                        size_blocks=px_to_block(e.size),
                        blend=e.blend)


def encode_schema(s: Schema, encode: CLIPEncoder) -> Embeds:
    return Embeds(background_embed=encode.prompt(s.background_prompt),
                  style_start_embed=encode.prompt(s.style_start_prompt),
                  style_end_embed=encode.prompt(s.style_end_prompt),
                  style_blend=s.style_blend,
                  entities=[encode_entity(e, encode) for e in s.entities])
