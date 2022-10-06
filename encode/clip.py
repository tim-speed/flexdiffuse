from typing import List
import torch
from transformers.models.clip.modeling_clip import CLIPModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer


class CLIPEncoder():
    def __init__(self, clip: CLIPModel, token: CLIPTokenizer) -> None:
        self.clip = clip
        self.token = token

    def prompt(self, prompt: str | List[str]) -> torch.Tensor:
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