from transformers import AutoConfig
import torch
import flashinfer
from flash_attn import flash_attn_with_kvcache
import math

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
            batch_size,
            max_length,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            max_length,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
   
    def gather_kv_incremental(self, indices: torch.LongTensor, offset:int):

        self.k_cache[:,:,offset:offset + len(indices), :,:] = self.k_cache[:,:,indices, :,:]
        self.v_cache[:,:,offset:offset + len(indices), :,:] = self.v_cache[:,:,indices, :,:]

        self.k_cache[:,:,offset + len(indices):, :,:] = 0.0
        self.v_cache[:,:,offset + len(indices):, :,:] = 0.0

        self.kv_offset = offset + len(indices)


    
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            storage_ids: torch.LongTensor=None
            ):

        new_kv_len = new_k_cache.shape[1] # [bsz, seq, num_heads, head_dim]
        if layer_idx == 0:
            self.kv_offset += new_kv_len
        self.k_cache[layer_idx][:, self.kv_offset - new_kv_len:self.kv_offset] = new_k_cache
        self.v_cache[layer_idx][:, self.kv_offset - new_kv_len:self.kv_offset] = new_v_cache
        return self.k_cache[layer_idx][:, :self.kv_offset], self.v_cache[layer_idx][:, :self.kv_offset]
    
    def compute_attention(self, 
        query_states :torch.Tensor,
        key_states :torch.Tensor, 
        value_states :torch.Tensor,
        layer_idx, 
        storage_ids :torch.Tensor=None,
        attention_mask :torch.Tensor=None):
        
        key_states, value_states = self.update_kv_cache(key_states, value_states, layer_idx, storage_ids)
        
        if attention_mask is not None:
            hidden_states = flashinfer.single_prefill_with_kv_cache(
                    q = query_states[0],
                    k = key_states[0],
                    v = value_states[0],
                    kv_layout="NHD",
                    custom_mask=attention_mask[:,:self.kv_offset],
                    allow_fp16_qk_reduction=True
                )
        else:
            # do not use attn mask
            # print(query_states.shape, key_states.shape, value_states.shape)
            hidden_states = flash_attn_with_kvcache(q=query_states, k_cache=key_states, v_cache=value_states, causal=True)
        
        return hidden_states
        
    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0
    
    def set_kv_len(self, kv_len :int):
            self.kv_offset = kv_len

    def get_kv_len(self):
        return self.kv_offset


class StaticKV_Cache:
    
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
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

    
    def gather_kv_incremental(self, indices: list[int], offset:int):

        self.k_cache[..., offset:offset + len(indices), :] = self.k_cache[..., indices, :]
        self.v_cache[..., offset:offset + len(indices), :] = self.v_cache[..., indices, :]

        self.k_cache[..., offset + len(indices):, :] = 0.0
        self.v_cache[..., offset + len(indices):, :] = 0.0

        self.kv_offset = offset + len(indices)


    
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            storage_ids: torch.LongTensor
            ):
        
        self.k_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_k_cache)
        self.v_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_v_cache)
        
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
        

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0
    
    def set_kv_len(self, kv_len :int):
            self.kv_offset = kv_len
    
    def compute_attention(self, 
        query_states :torch.Tensor,
        key_states :torch.Tensor, 
        value_states :torch.Tensor,
        layer_idx, 
        storage_ids :torch.Tensor,
        attention_mask :torch.Tensor):
        bsz, _, q_len, _ = query_states.shape
        
        key_states, value_states = self.update_kv_cache(key_states[0], value_states[0], layer_idx, storage_ids)        
        query_states = query_states[0]
        
        query_states = query_states.reshape(self.num_key_value_heads, q_len * self.num_key_value_groups, self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)
        mask = attention_mask[None,:,:].repeat(1, self.num_key_value_groups, 1)
        
        attn_weights.masked_fill_(~mask, torch.finfo(attn_weights.dtype).min)
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        hidden_states = torch.matmul(attn_weights, value_states)
        hidden_states = hidden_states.reshape(bsz, self.num_attention_heads, q_len, -1)
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        
        return hidden_states