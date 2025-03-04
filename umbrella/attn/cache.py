from transformers import AutoConfig
import torch
import flashinfer
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
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            max_length,
            config.num_key_value_heads,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            max_length,
            config.num_key_value_heads,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
   
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
            storage_ids: torch.LongTensor = None
            ):

        new_kv_len = storage_ids.shape[0] if storage_ids is not None else new_k_cache.shape[0]
        if layer_idx == 0:
            self.kv_offset += new_kv_len
        self.k_cache[layer_idx][self.kv_offset - new_kv_len:self.kv_offset] = new_k_cache
        self.v_cache[layer_idx][self.kv_offset - new_kv_len:self.kv_offset] = new_v_cache
        return self.k_cache[layer_idx][:self.kv_offset], self.v_cache[layer_idx][:self.kv_offset]
    
    def compute_attention(self, 
        query_states :torch.Tensor,
        key_states :torch.Tensor, 
        value_states :torch.Tensor,
        layer_idx, 
        storage_ids :torch.Tensor = None,
        attention_mask :torch.Tensor = None,
        logits_soft_cap = 0):
        
        key_states, value_states = self.update_kv_cache(key_states[0], value_states[0], layer_idx, storage_ids)
        
        if attention_mask is not None:
            hidden_states = flashinfer.single_prefill_with_kv_cache(
                    q = query_states[0],
                    k = key_states,
                    v = value_states,
                    kv_layout="NHD",
                    custom_mask=attention_mask[:,:self.kv_offset],
                    allow_fp16_qk_reduction=True,
                    logits_soft_cap = logits_soft_cap
                )
        
        else:
            hidden_states = flashinfer.single_prefill_with_kv_cache(
                    q = query_states[0],
                    k = key_states,
                    v = value_states,
                    kv_layout="NHD",
                    allow_fp16_qk_reduction=True,
                    logits_soft_cap = logits_soft_cap,
                    causal=True
                )
        return hidden_states
        
    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0
    
    def set_kv_len(self, kv_len :int):
            self.kv_offset = kv_len



class H2OCache:

    def __init__(self, 
        config :AutoConfig,
        batch_size :int = 1,
        kv_budget: int = 256,
        max_length :int = 256, 
        full_layers : list[int] = [0,1],
        device :str = 'cuda:0',
        dtype = torch.float16) -> None:
        self.config = config
        self.max_length = max_length
        self.kv_budget = kv_budget
        self.full_layers = full_layers
        self.device = device
        self.dtype = dtype
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.k_cache = [torch.zeros(
            config.num_key_value_heads,
            max_length if i in self.full_layers else self.kv_budget,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        ) for i in range(config.num_hidden_layers)]

        self.v_cache = [torch.zeros(
            config.num_key_value_heads,
            max_length if i in self.full_layers else self.kv_budget,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        ) for i in range(config.num_hidden_layers)]
        
        
        self.score = torch.zeros(
            config.num_hidden_layers,
            config.num_key_value_heads,
            self.kv_budget,
            device=self.device,
            dtype=torch.bfloat16
        )
        
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.decay = 0.8
        self.sink_tokens = 16
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
   

    def compute_attention(self, 
        query_states :torch.Tensor,
        key_states :torch.Tensor, 
        value_states :torch.Tensor,
        layer_idx, 
        storage_ids :torch.Tensor = None,
        attention_mask :torch.Tensor = None,
        logits_soft_cap = 0):
        
        bsz, q_len, _, _ = query_states.shape
        
        if layer_idx == 0:
                self.kv_offset += q_len
        
        if self.kv_offset <= self.kv_budget or layer_idx in self.full_layers:
            
            self.k_cache[layer_idx][:,self.kv_offset - q_len:self.kv_offset] = key_states[0].transpose(0,1)
            self.v_cache[layer_idx][:,self.kv_offset - q_len:self.kv_offset] = value_states[0].transpose(0,1)
            if q_len > 1 or layer_idx in self.full_layers:
                hidden_states = flashinfer.single_prefill_with_kv_cache(
                            q = query_states[0],
                            k = self.k_cache[layer_idx][:,:self.kv_offset],
                            v = self.v_cache[layer_idx][:,:self.kv_offset],
                            kv_layout="HND",
                            allow_fp16_qk_reduction=True,
                            logits_soft_cap = logits_soft_cap,
                            causal=True
                        )
            else:
                query_states = query_states.reshape(self.num_key_value_heads, q_len * self.num_key_value_groups, self.head_dim)
                attn_weights = torch.matmul(query_states, self.k_cache[layer_idx][:,:self.kv_offset].transpose(1, 2)) / math.sqrt(self.head_dim)
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                
                heuristic_score = attn_weights.sum(dim=-2)
                self.score[layer_idx][:,:self.kv_offset] = heuristic_score + self.score[layer_idx][:,:self.kv_offset] * self.decay
                hidden_states = torch.matmul(attn_weights, self.v_cache[layer_idx][:,:self.kv_offset])
                hidden_states = hidden_states.reshape(bsz, self.num_attention_heads, q_len, -1)
                hidden_states = hidden_states.transpose(1, 2).contiguous()
        else:   
                self.score[layer_idx][:,:self.sink_tokens] = torch.inf
                indices = self.score[layer_idx].argmin(dim=-1, keepdim=True)
                self.score[layer_idx].scatter_(dim=-1, index=indices, value=0)
                indices = indices[:,:,None].expand(self.num_key_value_heads, 1, self.head_dim)
                key_states = key_states.squeeze()[:,None,:]
                value_states = value_states.squeeze()[:,None,:]
                
                self.k_cache[layer_idx].scatter_(dim=-2, index=indices, src=key_states)
                self.v_cache[layer_idx].scatter_(dim=-2, index=indices, src=value_states)
                
                query_states = query_states.reshape(self.num_key_value_heads, q_len * self.num_key_value_groups, self.head_dim)
                attn_weights = torch.matmul(query_states, self.k_cache[layer_idx].transpose(1, 2)) / math.sqrt(self.head_dim)
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                
                heuristic_score = attn_weights.sum(dim=-2)

                self.score[layer_idx] = heuristic_score + self.score[layer_idx] * self.decay
                hidden_states = torch.matmul(attn_weights, self.v_cache[layer_idx])
                hidden_states = hidden_states.reshape(bsz, self.num_attention_heads, q_len, -1)
                hidden_states = hidden_states.transpose(1, 2).contiguous()

        
        return hidden_states
        
    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0
    
    def set_kv_len(self, kv_len :int):
            self.kv_offset = kv_len


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
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            config.num_key_value_heads,
            max_length,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            config.num_key_value_heads,
            max_length,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
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
