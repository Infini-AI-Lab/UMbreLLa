from abc import ABC, abstractmethod
import torch

class LLMBase(ABC):
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def alloc(self, **kwargs):
        
        raise NotImplementedError("Subclasses must implement the `alloc` method.")

    
    @abstractmethod
    def inference(self, 
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        storage_ids: torch.LongTensor):
        
        raise NotImplementedError("Subclasses must implement the `alloc` method.")
    
    