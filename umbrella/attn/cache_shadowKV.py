import torch.nn as nn
import torch
import math
import gc
from torch import nn
    
class ShadowKVCache:
    """ShadowKV, only for accuracy measurement and understanding, not for efficiency, please refer to ShadowKV_CPU for the efficient implementation"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
        rank=160,
        ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.rank = rank
        self.local_chunk = 4
        self.outlier_chunk = 48

        assert self.batch_size == 1, "ShadowKV class only supports batch_size=1, please use ShadowKV_CPU class for batch_size > 1"

        self.selected_chunk_idx = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget // self.chunk_size,
            device=self.device,
            dtype=torch.long
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )


        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"ShadowKV | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} |rank {self.rank} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    def get_svd(self, new_k_cache, layer_idx):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        if new_k_cache.shape[1] <= 32:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads*self.head_dim)
        else:
            # [bsz, prefill, 1024]
            k_cache = new_k_cache
        
        if layer_idx == 0:
            # init U, SV
            self.U = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.rank, device=self.device, dtype=self.dtype)
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.rank, self.head_dim, device=self.device, dtype=self.dtype)
        
        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1,2)
        # [bsz, 128k, 1024] --> [bsz, 128k, 160] [bsz, 160, 1024] (bsz, 8, 160, 128)
        self.U[layer_idx].copy_(u[:, :, :self.rank].to(self.dtype)) # [bsz, 128k, 160]
        self.SV[layer_idx].copy_(torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype).view(self.batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)) # [bsz, 8, 160, 128]
    
    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        if layer_idx == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device=self.device, dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device=self.device, dtype=torch.long)
        
        self.k_landmark[layer_idx].copy_(k_landmark.contiguous())
        self.k_landmark_idx[layer_idx].copy_(k_landmark_idx.contiguous())

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            query: torch.Tensor=None
            ):
        
        incoming = new_v_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        self.select_sets = self.sparse_budget // self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(self.batch_size, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(self.batch_size, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def get_retrieval_position_ids(self, layer_idx, query_states):
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1
        # print(query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim).shape, self.k_landmark[layer_idx].transpose(2, 3).shape)
        # [bsz, 8, 4, q_len, 128] * [bsz, 8, 128, chunks] --> [bsz, 8, 4, q_len, chunks]
        chunk_attn = torch.einsum('bhgqd,bhdc->bhgqc', query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), self.k_landmark[layer_idx].transpose(2, 3)).squeeze(2) / math.sqrt(128)
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype) # [bsz, 8, 4, q_len, chunks]
        chunk_attn = chunk_attn.sum(dim = -2) # [bsz, 8, 4, chunks]
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(256)]

        # use merged_results to gather the position_ids: [bsz, 8, select_sets] --> [bsz, 8, select_sets]
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]

        # this is chunk idx, which can be used to offload value cache and decide if the cache hits
        self.selected_chunk_idx[layer_idx].copy_(selected_chunks, non_blocking=True)

        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(self.batch_size, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

        return position_ids
        
    def get_value_cache(self, layer_idx, position_ids):
        # gather value cache
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        # gather key cache and rope them
        u = self.U[layer_idx] # [bsz, 128k, rank]
        sv = self.SV[layer_idx] # [bsz, 8, rank, 128]

        # indexing, [bsz, 8, sparse_budget, rank]
        index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1)) # [bsz, 8, sparse_budget, rank]
        u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1) # [bsz, 8, 128k, rank]
        U_head = torch.gather(u_expand, 2, index_expanded)

        # [bsz, 8, sparse_budget, rank] -matmul- [8, rank, 128] --> [bsz, 8, sparse_budget, 128]
        result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)

        # rope the key cache
        result = rope_func(result, position_ids)

        # send to buffer
        self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]
        self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_v_cache, non_blocking=True)
        self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_k_cache, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.selected_chunk_idx.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0
    
    def H2D(self):
        pass

    def get_kv_len(self):
        return self.kv_offset