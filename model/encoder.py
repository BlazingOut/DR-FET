import re
import numpy as np
from typing import List, Union

from tqdm import tqdm
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from abc import abstractmethod, ABC

def clean_text(text):
    # Replace underscores, hyphens, parentheses, and remove possessive 's with spaces
    cleaned = re.sub(r"[-_()（）]|('s\b)", " ", text)
    # Split into words and filter out empty strings
    return cleaned.split()

class BaseEmbedder(ABC):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        pass

    def embed_train(self):
        pass


class BertEmbed(BaseEmbedder):
    def __init__(self, model, pretrained_path: str, normalize: bool = True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.model = model.to(self.device)
        self.normalize = normalize

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling with attention mask
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        is_single = isinstance(texts, str)
        texts = [texts] if is_single else texts

        # Batch encode
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded_inputs['attention_mask'])
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.cpu().numpy()
        return embeddings[0] if is_single else embeddings


class E5Embed(BaseEmbedder):
    def __init__(self, model, pretrain_path, prefix='query: '):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        self.model = model.to(self.device)
        self.prefix = prefix  # Default is 'query: ', can be set to 'passage: '

    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Mask padding tokens with zeros
        masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed(self, texts: Union[str, List[str]]):
        is_single = isinstance(texts, str)
        texts = [texts] if is_single else texts

        # Add prefix to each text
        texts = [self.prefix + t for t in texts]

        # Encode in batches (can be optimized for larger batches)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            # embeddings = F.normalize(embeddings, p=2, dim=1)  # Optional L2 normalization

        embeddings = embeddings.cpu().numpy()
        return embeddings[0] if is_single else embeddings

class BGEEmbed(BaseEmbedder):
    def __init__(self,
                 model,
                 pretrain_path:str,
                 instruction: str = '为这个句子生成表示: ',
                 add_instruction: bool = True,
                 normalize: bool = False):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        self.instruction = instruction
        self.add_instruction = add_instruction
        self.normalize = normalize

    def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        is_single = isinstance(texts, str)
        texts = [texts] if is_single else texts

        if self.add_instruction:
            texts = [self.instruction + t for t in texts]

        encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_inputs)
            embeddings = model_output.last_hidden_state[:, 0]  # CLS pooling

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.cpu().numpy()
        return embeddings[0] if is_single else embeddings


def batch_encode(embedder,
                 texts: Union[List[str], str],
                 batch_size: int = 32,
                 show_progress: bool = False) -> np.ndarray:
    """
    Generic batch encoding function for embedder classes with embed() method.
    Compatible with embedder classes like BertEmbed, E5Embed, BGEEmbed.

    Args:
        embedder: Embedder instance implementing embed(texts: Union[str, List[str]])
        texts: List of texts or single text to encode
        batch_size: Size of each batch
        show_progress: Whether to show tqdm progress bar

    Returns:
        np.ndarray: shape = (N, hidden_size)
    """
    is_single = isinstance(texts, str)
    texts = [texts] if is_single else texts

    embeddings = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding", leave=False)

    for start_idx in iterator:
        batch_texts = texts[start_idx: start_idx + batch_size]
        batch_embeds = embedder.embed(batch_texts)
        embeddings.append(batch_embeds)

    all_embeddings = np.vstack(embeddings)
    return all_embeddings[0] if is_single else all_embeddings

def get_encoder(encoder_name, pretrained_model_path):
    embedder = AutoModel.from_pretrained(pretrained_model_path)
    if encoder_name == "bert":
        return BertEmbed(embedder, pretrained_model_path)
    elif encoder_name == "e5":
        return E5Embed(embedder, pretrained_model_path)
    elif encoder_name == "bge":
        return BGEEmbed(embedder, pretrained_model_path)
    else:
        raise ValueError(f"Encoder {encoder_name} not implemented")
