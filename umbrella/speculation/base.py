from abc import ABC, abstractmethod
import torch

class BaseEngine(ABC):
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def initialize(self):
        raise NotImplementedError
    
    @abstractmethod
    def verify(self):
        raise NotImplementedError
    
    @abstractmethod
    def build_tree(self):
        raise NotImplementedError
    
    @abstractmethod
    def prefill(self, text:str):
        raise NotImplementedError
    
    @abstractmethod
    def append(self, text:str):
        raise NotImplementedError
    
    @abstractmethod
    def _prefill(self, input_ids:torch.LongTensor):
        raise NotImplementedError
    
    @abstractmethod
    def _append(self, input_ids:torch.LongTensor):
        raise NotImplementedError
    
    
    @abstractmethod
    def speculative_decoding(self, max_new_tokens: int):
        raise NotImplementedError
    
    @abstractmethod
    def validate_status(self):
        raise NotImplementedError
    
    @abstractmethod
    def update_generation_args(self, **generation_args):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, **api_args):
        raise NotImplementedError
    