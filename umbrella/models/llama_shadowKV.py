################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

# Base LLM class

import torch
import torch.nn.functional as F
import time
import gc
from tqdm import tqdm

from flash_attn import flash_attn_with_kvcache
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import vllm

from .shadow_kv_ops import sample_token, layer_norm, minference_prefill_kernel, apply_rotary_pos_emb_cuda
# from .kv_cache import KV_Cache, ShadowKVCache, ShadowKVCache_CPU
from ..attn.cache import ShadowKVCache
from .base import LLMBase

Templates = {
    'base': "{ctx}",
    'llama-3': "<|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>{ctx}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    'yi': "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\n",
    'glm': "<|system|>\nYou are a helpful assistant\n<|user|> \n{ctx}<|assistant|>\n",
    'lwm': "You are a helpful assistant.\nUSER: {ctx}\nASSISTANT: Answer: ",
    'qwen': "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\n",
    'phi': "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\n{ctx}<|end|>\n<|assistant|>\n",
}

Chat_Templates = {
    'base': "{msg}",
    'llama-3': "<|start_header_id|>user<|end_header_id|>{msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    'yi': "<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n",
    'glm': "<|user|>\n{msg}<|assistant|>\n",
    'lwm': "\nUSER: {msg}\nASSISTANT: ",
    'qwen': "<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n",
    'phi': "<|user|>\n{msg}<|end|>\n<|assistant|>\n",
}

Prefix_Templates = {
    'base': "{ctx}",
    'llama-3': "<|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>{ctx}<|eot_id|><|start_header_id|>assistant<|end_header_id|>OK! I will help you with that. Please ask me anything.<|eot_id|>",
    'yi': "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\nOK! I will help you with that. Please ask me anything.\n",
    'glm': "<|system|>\nYou are a helpful assistant\n<|user|> \n{ctx}<|assistant|>\nOK! I will help you with that. Please ask me anything.\n",
}

class LLMBase_ShadowKV(LLMBase):
    def __str__(self) -> str:
        gpu_mem = f"{round(torch.cuda.memory_allocated(self.device) / 1024**3, 2)} GB / {round(torch.cuda.get_device_properties(self.device).total_memory / 1024**3, 2)} GB"
        return f"LLM: {self.model_name}, attn_mode: {self.attn_mode}, max_length: {self.max_length}, batch_size: {self.batch_size}, device: {self.device}, dtype: {self.dtype}, GPU mem: {gpu_mem}"

    def init_kv_cache(self, sparse_budget: int, rank: int, chunk_size: int, config):
        # if self.attn_mode == 'full':
        #     self.kv_cache = KV_Cache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        # elif self.attn_mode.lower() == 'shadowkv':
        #     self.kv_cache = ShadowKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        # elif self.attn_mode.lower() == 'shadowkv_cpu':
        #     self.kv_cache = ShadowKVCache_CPU(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        if self.attn_mode.lower() == 'shadowkv':
            self.kv_cache = ShadowKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

    def alloc(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the `alloc` method.")

    def print_kv_stats(self):
        self.kv_cache.print_stats()
    
    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        past_len = self.kv_cache.get_kv_len()
        position_ids = torch.arange(past_len, past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        return position_ids

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor = None,
            storage_ids: torch.LongTensor = None):

        hidden_states = F.embedding(input_ids, self.embed_tokens)

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)
        
        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
        
        if hidden_states.shape[1] > 16: # prefill
            hidden_states = hidden_states[:, -1:, :]
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        self.kv_cache.clear()
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))

        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"
        return logits

    @torch.inference_mode()
    def prefill_cont(self, input_ids: torch.LongTensor):
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))
        return logits
    
    def encode(self, text: str, template=None, truncation=False):
        if template == 'chat':
            text = self.chat_template.format(msg=text)
            input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if self.tokenizer.bos_token_id is not None:
                assert self.tokenizer.bos_token_id not in input_ids, f"bos_token_id found in input_ids"
            return input_ids
        if template == 'ctx':
            text = self.ctx_template.format(ctx=text)
        if template == 'prefix':
            text = self.prefix_template.format(ctx=text)
        input_ids = self.tokenizer(text, return_tensors="pt", truncation=truncation).input_ids.to(self.device)
        return input_ids

    @torch.inference_mode()
    def layer_compute(self, 
            buffer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor=None,
            storage_ids: torch.LongTensor=None):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        
        if isinstance(self.kv_cache, KV_Cache):
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
            key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
            
            if self.minference == True and q_len > 1:
                hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
            else:
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        elif isinstance(self.kv_cache, ShadowKVCache) or isinstance(self.kv_cache, ShadowKVCache_CPU):

            if q_len > 4*1024: # prefill
                # svd unrope key and save
                self.kv_cache.get_svd(key_states, layer_idx=layer_idx)
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                self.kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states[:, :, -1:])
                
                if self.minference == True:
                    hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
                else:
                    hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

            else: # decode
                # rope query and key
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

                # update kv cache to buffer
                self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)

                # get retrieval idx
                position_ids = self.kv_cache.get_retrieval_position_ids(layer_idx=layer_idx, query_states=query_states)

                # multi-stream
                curr_stream = torch.cuda.current_stream()
                get_value_stream = self.kv_cache.copy_stream

                with torch.cuda.stream(get_value_stream):
                    get_value_stream.wait_stream(curr_stream)
                    value_states = self.kv_cache.get_value_cache(layer_idx, position_ids)

                # gather key cache from GPU and RoPE it (should be hide by CPU offloading time)
                key_states = self.kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single, cos_sin_cache=self.cos_sin_cache)

                curr_stream.wait_stream(get_value_stream)

                # flash attention
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
        
        if bsz*q_len > 64*1024: # [bsz, seq, 128]
            output = torch.empty_like(hidden_states)
            prop_iter = bsz * q_len // (8*1024)
            prefill_chunk_size = bsz * q_len // prop_iter
            prefill_iter = (q_len + prefill_chunk_size - 1) // prefill_chunk_size
            for i in range(prefill_iter):
                start = i*prefill_chunk_size
                end = (i+1)*prefill_chunk_size
                output[:, start:end] = self.post_attention_compute(hidden_states[:, start:end], residual[:, start:end], buffer)
            
            hidden_states = output

        else:
            hidden_states = self.post_attention_compute(hidden_states, residual, buffer)
        
        return hidden_states

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool = False):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = 0.9, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """accuracy eval usage, not for throughput eval"""
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill(input_ids)
        else:
            if input_ids.size(1) + self.kv_cache.get_kv_len() >= self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill_cont(input_ids)
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        pos = 0
        generated_ids = []
        generated_ids.extend(next_token[0].tolist())
        
        self.kv_cache.H2D()

        if benchmark == True:
            start = time.time()
        
        while n < gen_len:
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.extend(next_token[0].tolist())
            if verbose == True:
                generated_text = (
                    self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    ).strip().split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            if next_token[0] == self.tokenizer.eos_token_id:
                break
            if self.tokenizer.decode(next_token[0]) == "<|eot_id|>": # llama-3
                break
            if self.tokenizer.decode(next_token[0]) == "<|im_end|>": # yi
                break
            if next_token[0] in [151329, 151336, 151338]: # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|endoftext|>": # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|end|>": # phi
                break

        if verbose == True and n!=0:
            print(" ".join(generated_text[pos:]), end=" ", flush=True)
        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s, {round(n / (end - start), 2)} tokens/s | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return [self.tokenizer.decode(generated_ids, skip_special_tokens=True)]
    
    @torch.inference_mode()
    def batch_prefill(self, input_ids: torch.Tensor, benchmark: bool = False):
        self.kv_cache.clear()
        batch_size = input_ids.size(0)
        
        assert batch_size == self.batch_size, f"batch_size mismatch, got {batch_size}, expected {self.batch_size}"
        
        if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
        
        logits = torch.zeros(batch_size, 1, self.vocab_size, device=self.device, dtype=torch.float32)

        if input_ids.shape[-1] > 120*1024 and input_ids.shape[-1] < 200*1024:
            T = 8
        else:
            T = 4
        # for bsz in range(0, batch_size, T):
        for bsz in tqdm(range(0, batch_size, T), desc=f"Prefilling (batch size={batch_size})"):
            req_input_ids = input_ids[bsz:bsz+T]
            logits[bsz:bsz+T].copy_(self.inference(input_ids=req_input_ids, position_ids=self.get_ctx(req_input_ids)))
        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"

        return logits


    @torch.inference_mode()
    def warmup(self):

        a = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        b = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        for _ in range(100):
            torch.bmm(a, b)
        del a, b

        print("Warmup done")

    @torch.inference_mode()
    def batch_generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = -1, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """throughput eval usage"""
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.batch_prefill(input_ids)
        else:
            logits = self.prefill_cont(input_ids)
        
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        generated_ids = []
        generated_ids.append(next_token[:, -1].tolist())
        
        self.kv_cache.H2D()
        self.warmup()

        if benchmark == True:
            start = time.time()
        
        while n < gen_len:
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.append(next_token[:, -1].tolist())

        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s | Throughput: {round(self.batch_size * n / (end - start), 2)} tokens/s, Latency: {round((end - start)*1000 / n, 2)} ms/step | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        generated_ids = torch.LongTensor(generated_ids).t().tolist()

        if benchmark == True:
            return self.decode(generated_ids, skip_special_tokens=True), self.batch_size * n / (end - start)

        return self.decode(generated_ids, skip_special_tokens=True)
    
class LlamaLayer:
    def __init__(self, layer_idx) -> None:
        
        self.wqkv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.gate_up_proj :torch.Tensor = None 
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx

    def init_parameters(self, hf_layer: LlamaDecoderLayer):

        self.wqkv :torch.Tensor= torch.cat((hf_layer.self_attn.q_proj.weight.detach(), hf_layer.self_attn.k_proj.weight.detach(), hf_layer.self_attn.v_proj.weight.detach()), dim=0)
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()
        self.q_size = hf_layer.self_attn.q_proj.weight.shape[0]
        self.kv_size = hf_layer.self_attn.k_proj.weight.shape[0]

        self.gate_up_proj = torch.cat((hf_layer.mlp.gate_proj.weight.detach(), hf_layer.mlp.up_proj.weight.detach()), dim=0)
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon
    
    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wqkv = self.wqkv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_up_proj = self.gate_up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

class Llama(LLMBase_ShadowKV):
    def __init__(self, 
        model_name: str = "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        batch_size :int = 1,
        max_length :int = 64*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        attn_mode: str = 'full',
        sparse_budget: int = 2048,
        rank=160,
        chunk_size=8,
        minference=False) -> None:
        
        # assert batch_size == 1, "Batch size must be 1"
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = LlamaConfig.from_pretrained(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.vocab_size = self.config.vocab_size

        self.init_parameters()
        self.attn_mode = attn_mode
        self.minference = minference

        if 'llama-3' in model_name.lower():
            self.ctx_template = Templates['llama-3']
            self.chat_template = Chat_Templates['llama-3']
            self.prefix_template = Prefix_Templates['llama-3']
        elif 'yi' in model_name.lower():
            self.ctx_template = Templates['yi']
            self.chat_template = Chat_Templates['yi']
            self.prefix_template = Prefix_Templates['yi']
        else:
            raise ValueError(f"Invalid model name {model_name}")

        self.init_kv_cache(sparse_budget, rank, chunk_size, self.config)

        if self.minference:
            import json
            self.minference_parttern = []
            for layer_idx in range(self.num_layers):
                self.minference_parttern.append({int(ii): jj for ii, jj in json.load(open(MODEL2PATH[self.model_name]))[layer_idx].items()})

    def alloc(self, **kwargs):
        pass

    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        t = torch.arange(self.max_length + 1024, device=self.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb_cuda(x, self.cos_sin_cache, position_ids)

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        vllm._custom_ops.rotary_embedding(position_ids, q, k, 128, self.cos_sin_cache, True)
        bsz = q.shape[0]
        q = q.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k

    def init_parameters(self):
        hf_model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        try:
            cos_cache = hf_model.model.layers[0].self_attn.rotary_emb.cos_cached[:self.max_length+1024].to(self.device).to(self.dtype)
            sin_cache = hf_model.model.layers[0].self_attn.rotary_emb.sin_cached[:self.max_length+1024].to(self.device).to(self.dtype)
        except:
            cos_cache, sin_cache = self._set_cos_sin_cache(hf_model.model.layers[0].self_attn.rotary_emb.inv_freq.to(self.device))
        self.cos_sin_cache = torch.cat((cos_cache[:, :64], sin_cache[:, :64]), dim=-1)
        
        del cos_cache, sin_cache

        self.layers :list[LlamaLayer] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = LlamaLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: LlamaLayer,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        qkv = F.linear(hidden_states, buffer.wqkv)
        query_states, key_states, value_states = qkv.split([buffer.q_size, buffer.kv_size, buffer.kv_size], dim=-1)

        return query_states, key_states, value_states.view(value_states.shape[0], -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    def clear(self):
        self.kv_cache.clear()

    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: LlamaLayer
    ):  
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        
        hidden_states = F.linear(hidden_states, buffer.gate_up_proj)
        d = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        vllm._custom_ops.silu_and_mul(out, hidden_states)
        
        hidden_states = F.linear(out, buffer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states
