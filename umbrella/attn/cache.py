from transformers import AutoConfig
import torch

class KV_Cache:

    def __init__(self, 
        config :AutoConfig,
        batch_size :int = 1,
        max_length :int = 256, 
        device :str = 'cuda:0',
        dtype = torch.float16) -> None:
        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            max_length,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            max_length,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
   
    def gather_kv_incremental(self, indices: torch.LongTensor, offset:int):

        self.k_cache[:,offset:offset + len(indices), :,:] = self.k_cache[:,indices, :,:]
        self.v_cache[:,offset:offset + len(indices), :,:] = self.v_cache[:,indices, :,:]

        self.k_cache[:,offset + len(indices):, :,:] = 0.0
        self.v_cache[:,offset + len(indices):, :,:] = 0.0

        self.kv_offset = offset + len(indices)


    
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            storage_ids: torch.LongTensor
            ):

        new_kv_len = storage_ids.shape[0]
        if layer_idx == 0:
            self.kv_offset += new_kv_len
        self.k_cache[layer_idx][self.kv_offset - new_kv_len:self.kv_offset] = new_k_cache
        self.v_cache[layer_idx][self.kv_offset - new_kv_len:self.kv_offset] = new_v_cache
        return self.k_cache[layer_idx][:self.kv_offset], self.v_cache[layer_idx][:self.kv_offset]
    

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0
    
    def set_kv_len(self, kv_len :int):
            self.kv_offset = kv_len