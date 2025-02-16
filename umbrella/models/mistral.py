from transformers import MistralForCausalLM, MistralConfig, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import gc
import flashinfer
from ..attn.cache import KV_Cache, StaticKV_Cache
from .mistral_layer import MistralLayer, MistralAwqLayer, MistralPackedLayer
from .base import LLMBase
from .model_utils import apply_rotary_pos_emb, layer_norm, capture_graph
from tqdm import tqdm

class Mistral(LLMBase):
    def __init__(self,
        model_name: str,
        batch_size: int=1, 
        max_length: int=256, 
        device: str = "cuda:0", 
        dtype = torch.float16) -> None:

        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype 
        self.config = MistralConfig.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = getattr(self.config, 'head_dim', self.config.hidden_size // self.config.num_attention_heads)
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.eos_tokens = self.config.eos_token_id if (isinstance(self.config.eos_token_id, list)) else [self.config.eos_token_id]

    def alloc(self, **kwargs):
        
        self.kv_cache = KV_Cache(self.config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        hf_model = MistralForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        if self.config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        
        self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(self.device)
        self.attention_scaling = hf_model.model.rotary_emb.attention_scaling
        position_ids = torch.arange(0, self.max_length).unsqueeze(0).to(self.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[0]
        self.sin_cache = emb.sin()[0]
        self.cos_cache = self.cos_cache * self.attention_scaling
        self.sin_cache = self.sin_cache * self.attention_scaling
        self.cos_cache = self.cos_cache.to(self.dtype)
        self.sin_cache = self.sin_cache.to(self.dtype)
        
        self.layers :list[MistralLayer] = []
        
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = MistralLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.to(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()
            
        self.num_layers = len(self.layers)

    @torch.inference_mode()
    def layer_compute(self, 
            buffer: MistralLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor, 
            attention_mask: torch.FloatTensor,
            storage_ids: torch.LongTensor):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        query_states = F.linear(hidden_states, buffer.wq)
        key_states = F.linear(hidden_states, buffer.wk)
        value_states = F.linear(hidden_states, buffer.wv)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, self.cos_cache, self.sin_cache, position_ids)
        hidden_states = self.kv_cache.compute_attention(
            query_states, key_states, value_states, layer_idx, storage_ids, attention_mask
        )
        hidden_states = hidden_states.reshape(bsz, q_len, self.head_dim * self.num_heads)
        
        hidden_states = F.linear(hidden_states, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        up = F.linear(hidden_states, buffer.up_proj)
        gate = F.linear(hidden_states, buffer.gate_proj)
        gate = F.silu(gate)
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, buffer.down_proj)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            storage_ids: torch.LongTensor):
        
        hidden_states = F.embedding(input_ids, self.embed_tokens)  
        for idx in range(self.num_layers):
                hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, attention_mask, storage_ids)
        
        b, s, h = hidden_states.shape

        hidden_states = hidden_states.reshape(b * s, h)
        hidden_states = flashinfer.rmsnorm(hidden_states, self.norm_weight, self.norm_variance_epsilon)
        hidden_states = hidden_states.reshape(b, s, h)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    def gather_kv_incremental(self, indices: torch.LongTensor, offset:int):
        
        self.kv_cache.gather_kv_incremental(indices=indices, offset=offset)
    
    def clear(self):
        
        self.kv_cache.clear()


class MistralOffload(Mistral):
    def __init__(self, model_name, batch_size = 1, max_length = 256, device = 'cuda:0', dtype=torch.float16):
        super().__init__(model_name, batch_size, max_length, device, dtype)
        self.load_stream = torch.cuda.Stream(device=device)
    
    def alloc(self, **kwargs):
        
        
        self.num_cache_layers = kwargs["num_cache_layers"] if 'num_cache_layers' in kwargs else 0
        self.kv_cache = KV_Cache(self.config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        hf_model = MistralForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        if self.config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        
        self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(self.device)
        self.attention_scaling = hf_model.model.rotary_emb.attention_scaling
        position_ids = torch.arange(0, self.max_length).unsqueeze(0).to(self.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[0]
        self.sin_cache = emb.sin()[0]
        self.cos_cache = self.cos_cache * self.attention_scaling
        self.sin_cache = self.sin_cache * self.attention_scaling
        self.cos_cache = self.cos_cache.to(self.dtype)
        self.sin_cache = self.sin_cache.to(self.dtype)
        
        self.layers :list[MistralLayer] = []
        
        for idx, hf_layer in tqdm(enumerate(hf_model.model.layers), desc="initial offloaded model"):
            layer = MistralLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            if idx < self.num_cache_layers:
                layer.to(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()
            
        self.num_layers = len(self.layers)
        assert self.num_layers % 2 == 0
        self.buffer = [MistralLayer(-1, self.device) for _ in range(2)]
        self.buffer[0].alloc_space(self.layers[0], self.device)
        self.buffer[1].alloc_space(self.layers[0], self.device)
    
    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            storage_ids: torch.LongTensor):
        
        hidden_states = F.embedding(input_ids, self.embed_tokens)
        if self.buffer[0].layer_idx != 0:
            self.buffer[0].copy(self.layers[0])
            torch.cuda.synchronize()
        for idx in range(self.num_layers):
            with torch.cuda.stream(self.load_stream):
                self.buffer[(idx + 1) % 2].copy(self.layers[(idx + 1)% self.num_layers])
            
            hidden_states = self.layer_compute(self.buffer[idx % 2], idx, hidden_states, position_ids, attention_mask, storage_ids)
            torch.cuda.synchronize()
        b, s, h = hidden_states.shape

        hidden_states = hidden_states.reshape(b * s, h)
        hidden_states = flashinfer.rmsnorm(hidden_states, self.norm_weight, self.norm_variance_epsilon)
        hidden_states = hidden_states.reshape(b, s, h)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits


class MistralAwq(Mistral):
    def alloc(self, **kwargs):
        
        self.kv_cache = KV_Cache(self.config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        
        hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        if self.config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(self.device)
        self.attention_scaling = hf_model.model.rotary_emb.attention_scaling
        position_ids = torch.arange(0, self.max_length).unsqueeze(0).to(self.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[0]
        self.sin_cache = emb.sin()[0]
        self.cos_cache = self.cos_cache * self.attention_scaling
        self.sin_cache = self.sin_cache * self.attention_scaling
        self.cos_cache = self.cos_cache.to(self.dtype)
        self.sin_cache = self.sin_cache.to(self.dtype)
        self.layers :list[MistralAwqLayer] = []
        
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = MistralAwqLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.to(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()
        self.num_layers = len(self.layers)
       
        
    @torch.inference_mode()
    def layer_compute(self, 
            buffer: MistralAwqLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor, 
            attention_mask: torch.FloatTensor,
            storage_ids: torch.LongTensor):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        query_states = buffer.wq.apply(hidden_states)
        key_states = buffer.wk.apply(hidden_states)
        value_states = buffer.wv.apply(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, self.cos_cache, self.sin_cache, position_ids)

        hidden_states = self.kv_cache.compute_attention(
            query_states, key_states, value_states, layer_idx, storage_ids, attention_mask
        )
        
        hidden_states = hidden_states.reshape(bsz, q_len, self.head_dim * self.num_heads)
        
        hidden_states = buffer.wo.apply(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        up = buffer.up_proj.apply(hidden_states)
        gate = buffer.gate_proj.apply(hidden_states)
        gate = F.silu(gate)
        hidden_states = gate * up
        hidden_states = buffer.down_proj.apply(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            storage_ids: torch.LongTensor):
        
        hidden_states = F.embedding(input_ids, self.embed_tokens) 
        
        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, attention_mask, storage_ids)
            
        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h)
        hidden_states = flashinfer.rmsnorm(hidden_states, self.norm_weight, self.norm_variance_epsilon)
        hidden_states = hidden_states.reshape(b, s, h)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits


class MistralAwqOffload(MistralOffload):
    def alloc(self, **kwargs):
        
        self.num_cache_layers = kwargs["num_cache_layers"] if 'num_cache_layers' in kwargs else 0
        self.kv_cache = KV_Cache(self.config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        
        hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        if self.config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(self.device)
        self.attention_scaling = hf_model.model.rotary_emb.attention_scaling
        position_ids = torch.arange(0, self.max_length).unsqueeze(0).to(self.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[0]
        self.sin_cache = emb.sin()[0]
        self.cos_cache = self.cos_cache * self.attention_scaling
        self.sin_cache = self.sin_cache * self.attention_scaling
        self.cos_cache = self.cos_cache.to(self.dtype)
        self.sin_cache = self.sin_cache.to(self.dtype)
        self.layers :list[MistralAwqLayer] = []
        
        for idx, hf_layer in tqdm(enumerate(hf_model.model.layers), desc="initial offloaded model"):
            layer = MistralAwqLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            if idx < self.num_cache_layers:
                layer.to(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()
        self.num_layers = len(self.layers)
        assert self.num_layers % 2 == 0
        self.buffer = [MistralAwqLayer(-1, self.device) for _ in range(2)]
        self.buffer[0].alloc_space(self.layers[0], self.device)
        self.buffer[1].alloc_space(self.layers[0], self.device)
        
    @torch.inference_mode()
    def layer_compute(self, 
            buffer: MistralAwqLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor, 
            attention_mask: torch.FloatTensor,
            storage_ids: torch.LongTensor):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        query_states = buffer.wq.apply(hidden_states)
        key_states = buffer.wk.apply(hidden_states)
        value_states = buffer.wv.apply(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, self.cos_cache, self.sin_cache, position_ids)
        hidden_states = self.kv_cache.compute_attention(
            query_states, key_states, value_states, layer_idx, storage_ids, attention_mask
        )
        hidden_states = hidden_states.reshape(bsz, q_len, self.head_dim * self.num_heads)
        
        hidden_states = buffer.wo.apply(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        up = buffer.up_proj.apply(hidden_states)
        gate = buffer.gate_proj.apply(hidden_states)
        gate = F.silu(gate)
        hidden_states = gate * up
        hidden_states = buffer.down_proj.apply(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class MistralCudagraph(Mistral):
    def __init__(self, model_name, batch_size = 1, max_length = 256, device = 'cuda:0', dtype=torch.float16):
        super().__init__(model_name, batch_size, max_length, device, dtype)
    
        self.callables = {}
        self.mempool = None

    def alloc(self, **kwargs):
        
        exit_layer = kwargs.pop("exit_layer", -1)
        self.kv_cache = StaticKV_Cache(self.config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        hf_model = MistralForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        if self.config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        
        self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(self.device)
        self.attention_scaling = hf_model.model.rotary_emb.attention_scaling
        position_ids = torch.arange(0, self.max_length).unsqueeze(0).to(self.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[0]
        self.sin_cache = emb.sin()[0]
        self.cos_cache = self.cos_cache * self.attention_scaling
        self.sin_cache = self.sin_cache * self.attention_scaling
        self.cos_cache = self.cos_cache.to(self.dtype)
        self.sin_cache = self.sin_cache.to(self.dtype)
        
        self.layers :list[MistralPackedLayer] = []
        
        for idx, hf_layer in enumerate(hf_model.model.layers):
            if exit_layer > 0 and idx >= exit_layer:
                break
            layer = MistralPackedLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.to(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()
            
        self.num_layers = len(self.layers)
    
    @torch.inference_mode()
    def layer_compute(self, 
            buffer: MistralPackedLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor, 
            attention_mask: torch.FloatTensor,
            storage_ids: torch.LongTensor):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        qkv = F.linear(hidden_states, buffer.wqkv)
        query_states = qkv[...,:self.hidden_size]
        key_states = qkv[...,self.hidden_size:self.hidden_size + self.head_dim * self.num_key_value_heads]
        value_states = qkv[...,self.hidden_size + self.head_dim * self.num_key_value_heads:]
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, self.cos_cache, self.sin_cache, position_ids, unsqueeze_dim=1)       
        hidden_states = self.kv_cache.compute_attention(
            query_states, key_states, value_states, layer_idx, storage_ids, attention_mask
        )
        
        hidden_states = hidden_states.reshape(bsz, q_len, self.head_dim * self.num_heads)
        hidden_states = F.linear(hidden_states, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        up = F.linear(hidden_states, buffer.up_proj)
        gate = F.linear(hidden_states, buffer.gate_proj)
        gate = F.silu(gate)
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, buffer.down_proj)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    
    @torch.inference_mode()
    def initialize_cuda_graph(self, 
            decoding_seqlens :list[int],
            n_warmups=12):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        for decoding_seqlen in decoding_seqlens:
            if decoding_seqlen not in self.callables:
                self.callables[decoding_seqlen] = capture_graph(
                    llm=self,
                    decoding_seqlen=decoding_seqlen,
                    mempool=self.mempool,
                    n_warmups=n_warmups
                )
        self.clear()
        
    @torch.inference_mode()
    def graph_inference(self,
            input_ids: torch.LongTensor, 
            storage_ids :torch.LongTensor,
            position_ids = None,
            attention_mask = None,
            ):
            dec_length = input_ids.shape[1]
            if dec_length in self.callables.keys():
                logits = self.callables[dec_length](input_ids, storage_ids, position_ids, attention_mask)
            else:
                logits = self.inference(input_ids, position_ids, attention_mask, storage_ids)
            return logits