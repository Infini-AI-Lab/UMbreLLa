import torch
from ..models import AutoModelLM
from transformers import AutoTokenizer, GenerationConfig
from .speculation_utils import (
make_causal_mask, 
find_first_element_position, 
apply_repetition_penalty, 
apply_topk,
is_sentence_complete_regex
)
import time
import flashinfer
from ..logging_config import setup_logger
from ..utils import TextColors
from .base import BaseEngine
logger = setup_logger()

class AREngine(BaseEngine):
    def __init__(self,
        model_name: str,
        dtype=torch.float16,
        device :str = 'cuda:0',
        **kwargs
        ) -> None:

        super().__init__()

        self.model_name = model_name
        self.dtype = dtype
        self.device = device

        self.max_length = kwargs.pop("max_length", 8192)
        self.safe_buffer = kwargs.pop("safe_buffer", 8)
        self.temperature = kwargs.pop("temperature", 0.0)
        self.topp = kwargs.pop("topp", 0.9)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.topk = kwargs.pop("topk", 32)
        self.num_beams = kwargs.pop("num_beams", 24)
        self.offload = kwargs.pop("offload", False)
        self.config = kwargs

    def initialize(self):

        self.tokens = torch.zeros(1, self.max_length, device=self.device).long()
        self.model = AutoModelLM.from_pretrained(
                    model_name=self.model_name, offload=self.offload, batch_size=1, 
                    max_length=self.max_length, device=self.device,
                    dtype=self.dtype)

        self.model.alloc(**self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.vocab_size = self.model.config.vocab_size
        self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.eos_tokens = self.generation_config.eos_token_id if (isinstance(self.generation_config.eos_token_id, list)) else [self.generation_config.eos_token_id]