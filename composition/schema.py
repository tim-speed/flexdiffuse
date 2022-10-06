from dataclasses import dataclass
import json
from typing import List, Tuple


@dataclass
class EntitySchema():
    prompt: str
    offset: Tuple[int, int]
    size: Tuple[int, int]
    blend: float = 0.8
    # TODO: Shape / mask


@dataclass
class Schema():
    background_prompt: str
    style_start_prompt: str
    style_end_prompt: str
    style_blend: Tuple[float, float]
    entities: List[EntitySchema]

    def json(self) -> str:
        s = self.__dict__.copy()
        s['entities'] = [e.__dict__ for e in self.entities]
        return json.dumps(s)