from transformers import AutoConfig
import torch
import flashinfer
import math


def mha_flash(q, k, v, kv_layout, attn_mask):
    return flashinfer.single_prefill_with_kv_cache(
        q=q,
        k=k,
        v=v,
        kv_layout=kv_layout,
        custom_mask=attn_mask,
        allow_fp16_qk_reduction=True
    )

def mha(q, k, v, attn_mask):
    """
    Args:
        q (torch.Tensor): Query tensor of shape (query_len, q_head, head_dim)
        k (torch.Tensor): Key tensor of shape (kv_len, kv_head, head_dim)
        v (torch.Tensor): Value tensor of shape (kv_len, kv_head, head_dim)
        attn_mask (torch.Tensor): Value tensor of shape (q_len, kv_len)

    Returns:

    """
    q_len, q_head, head_dim = q.shape
    kv_len, kv_head, _ = k.shape
    assert (q_head % kv_head == 0)
    num_kv_groups = q_head // kv_head

    # Step 1: Reshape Q for GQA
    # (q_len, q_head, head_dim) -> (kv_head, num_kv_groups, q_len, head_dim)
    q = q.transpose(0,1).reshape(kv_head, num_kv_groups, q_len, head_dim)
    k = k.transpose(0,1)
    v = v.transpose(0,1)
    # Step 2: Compute Attention Scores (kv_head, num_kv_groups, q_len, kv_len)
    attn_scores = torch.einsum('hgld,hmd->hglm', q, k) / math.sqrt(head_dim)

    # Step 3: Apply Attention Mask
    # (kv_head, num_groups, q_len, kv_len)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(kv_head, num_kv_groups, -1, -1)
    attn_scores.masked_fill_(~attn_mask, torch.finfo(attn_scores.dtype).min)

    # Step 4: Compute Attention Weights
    # (kv_head, num_kv_groups, q_len, kv_len)
    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

    # Step 5: Compute Context Vector
    # (kv_head, num_kv_groups, q_len, head_dim)
    hidden_states = torch.einsum('hglm,hmd->hgld', attn_weights, v)
    hidden_states = hidden_states.reshape(q_head, q_len, head_dim)
    hidden_states = hidden_states.transpose(0, 1).contiguous()
    return hidden_states


class KV_Cache:
    def __init__(self,
                 config: AutoConfig,
                 batch_size: int = 1,
                 max_length: int = 256,
                 device: str = 'cuda:0',
                 dtype=torch.float16) -> None:
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

    def gather_kv_incremental(self, indices: torch.LongTensor, offset: int):
        self.k_cache[:, offset:offset + len(indices), :, :] = self.k_cache[:, indices, :, :]
        self.v_cache[:, offset:offset + len(indices), :, :] = self.v_cache[:, indices, :, :]

        self.k_cache[:, offset + len(indices):, :, :] = 0.0
        self.v_cache[:, offset + len(indices):, :, :] = 0.0

        self.kv_offset = offset + len(indices)

    def update_kv_cache(self,
                        new_k_cache: torch.Tensor,
                        new_v_cache: torch.Tensor,
                        layer_idx: int,
                        storage_ids: torch.LongTensor
                        ):
        new_kv_len = storage_ids.shape[0]
        if layer_idx == 0:
            self.kv_offset += new_kv_len
        self.k_cache[layer_idx][self.kv_offset - new_kv_len:self.kv_offset] = new_k_cache
        self.v_cache[layer_idx][self.kv_offset - new_kv_len:self.kv_offset] = new_v_cache
        return self.k_cache[layer_idx][:self.kv_offset], self.v_cache[layer_idx][:self.kv_offset]

    def compute_attention(self,
                          query_states: torch.Tensor,
                          key_states: torch.Tensor,
                          value_states: torch.Tensor,
                          layer_idx,
                          storage_ids: torch.Tensor,
                          attention_mask: torch.Tensor):
        key_states, value_states = self.update_kv_cache(key_states[0], value_states[0], layer_idx, storage_ids)
        hidden_states = mha_flash(query_states[0], key_states, value_states, "NHD", attention_mask[:, :self.kv_offset])

        return hidden_states

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0

    def set_kv_len(self, kv_len: int):
        self.kv_offset = kv_len


class StaticKV_Cache:
    def __init__(self,
                 config: AutoConfig,
                 batch_size: int = 1,
                 max_length: int = 256,
                 device: str = 'cuda:0',
                 dtype=torch.float16) -> None:
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

    def gather_kv_incremental(self, indices: list[int], offset: int):
        self.k_cache[..., offset:offset + len(indices), :] = self.k_cache[..., indices, :]
        self.v_cache[..., offset:offset + len(indices), :] = self.v_cache[..., indices, :]

        self.k_cache[..., offset + len(indices):, :] = 0.0
        self.v_cache[..., offset + len(indices):, :] = 0.0

        self.kv_offset = offset + len(indices)

    def update_kv_cache(self,
                        new_k_cache: torch.Tensor,
                        new_v_cache: torch.Tensor,
                        layer_idx: int,
                        storage_ids: torch.LongTensor
                        ):
        self.k_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_k_cache)
        self.v_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_v_cache)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0

    def set_kv_len(self, kv_len: int):
        self.kv_offset = kv_len

    def compute_attention(self,
                          query_states: torch.Tensor,
                          key_states: torch.Tensor,
                          value_states: torch.Tensor,
                          layer_idx,
                          storage_ids: torch.Tensor,
                          attention_mask: torch.Tensor):
        bsz, _, q_len, _ = query_states.shape

        key_states, value_states = self.update_kv_cache(key_states[0], value_states[0], layer_idx, storage_ids)
        query_states = query_states[0]

        query_states = query_states.reshape(self.num_key_value_heads, q_len * self.num_key_value_groups, self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)
        mask = attention_mask[None, :, :].repeat(1, self.num_key_value_groups, 1)

        attn_weights.masked_fill_(~mask, torch.finfo(attn_weights.dtype).min)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        hidden_states = torch.matmul(attn_weights, value_states)
        hidden_states = hidden_states.reshape(bsz, self.num_attention_heads, q_len, -1)
        hidden_states = hidden_states.transpose(1, 2).contiguous()

        return hidden_states


class SlidingWindowKV_Cache:
    def __init__(self,
                 config: AutoConfig,
                 batch_size: int = 1,
                 max_length: int = 256,
                 device: str = 'cuda:0',
                 dtype=torch.float16) -> None:

        self.config = config
        self.max_length = max_length
        self.window_size = self.config.window_size
        self.device = device
        self.dtype = dtype
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        # initializing Key-Value Cache
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
        self.kv_offset = 0

    def update_kv_cache(self,
                        new_k_cache: torch.Tensor,
                        new_v_cache: torch.Tensor,
                        layer_idx: int,
                        storage_ids: torch.LongTensor
                        ):

        new_kv_len = storage_ids.shape[0]
        # # calculating new offset
        need_len = new_kv_len + min(self.kv_offset, self.window_size)
        if need_len > self.max_length:
            raise ValueError(f'now kv_offset {self.kv_offset} new_kv_len {new_kv_len} need_len {need_len} exceeds max_length {self.max_length}')    

        # update KV Cache
        if layer_idx == 0:
            self.kv_offset += new_kv_len
            if self.kv_offset >= self.max_length:
                # print(f'layer idx need shift data, {self.kv_offset}')
                self.k_cache[:, :self.window_size, :, :] = self.k_cache[:, self.kv_offset - self.window_size: self.kv_offset, :, :]
                self.v_cache[:, :self.window_size, :, :] = self.v_cache[:, self.kv_offset - self.window_size: self.kv_offset, :, :]
                self.kv_offset = self.window_size + new_kv_len
            # print(f'layer_idx, add, {self.kv_offset}')
        self.k_cache[layer_idx][self.kv_offset - new_kv_len:self.kv_offset] = new_k_cache
        self.v_cache[layer_idx][self.kv_offset - new_kv_len:self.kv_offset] = new_v_cache
        start_idx = max(0, self.kv_offset - (self.window_size + new_kv_len))
        return self.k_cache[layer_idx][start_idx:self.kv_offset], self.v_cache[layer_idx][start_idx:self.kv_offset]

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0

    def set_kv_len(self, kv_len: int):
        self.kv_offset = kv_len

    def gather_kv_incremental(self, indices: torch.LongTensor, offset: int):
        self.k_cache[:, offset:offset + len(indices), :, :] = self.k_cache[:, indices, :, :]
        self.v_cache[:, offset:offset + len(indices), :, :] = self.v_cache[:, indices, :, :]

        self.k_cache[:, offset + len(indices):, :, :] = 0.0
        self.v_cache[:, offset + len(indices):, :, :] = 0.0

        self.kv_offset = offset + len(indices)

    def compute_attention(self,
                          query_states: torch.Tensor,
                          key_states: torch.Tensor,
                          value_states: torch.Tensor,
                          layer_idx,
                          storage_ids: torch.Tensor,
                          attention_mask: torch.Tensor):
        """
        Computes the attention output using a sliding window mechanism.

        This function updates the KV cache, constructs the appropriate attention mask that considers:
        - Sliding window attention (limits visible tokens to a fixed window size)
        - Original `attention_mask` (to handle padding tokens)
        - Causal Mask (ensures auto-regressive decoding)

        Args:
            query_states (torch.Tensor): Query tensor of shape (batch, num_heads, query_len, head_dim)
            key_states (torch.Tensor): Key tensor of shape (batch, num_heads, kv_len, head_dim)
            value_states (torch.Tensor): Value tensor of shape (batch, num_heads, kv_len, head_dim)
            layer_idx (int): Current layer index in the transformer model.
            storage_ids (torch.Tensor): Index positions to store the new KV cache.
            attention_mask (torch.Tensor): Casual mask. (q_len, squence_len)

        Returns:
            hidden_states (torch.Tensor): Output hidden states after applying attention.
        """

        # Step 1: Update KV Cache (keep only the latest window_size tokens)
        key_states, value_states = self.update_kv_cache(key_states[0], value_states[0], layer_idx, storage_ids)
        q_len = storage_ids.shape[0]
        kv_len = key_states.shape[0]
  
        # Step 2: Generate Sliding Window Attention Mask
        # Create a 2D attention mask where each query position can attend only to the latest `window_size` tokens
        query_indices = (kv_len - q_len) + torch.arange(q_len, device=self.device).unsqueeze(1)  # Shape: (q_len, 1)
        kv_indices = torch.arange(kv_len, device=self.device).unsqueeze(0)  # Shape: (1, kv_offset)
        diff = query_indices - kv_indices
        # Compute boolean mask: True if kv_index is within `window_size` of the query_index
        attn_mask = torch.logical_and(diff <= self.window_size, diff >= 0) # (q_len, kv_offset)
        # Step 3: Compute Attention Both are ok.
        # hidden_states = mha(query_states[0], key_states, value_states, attn_mask)
        hidden_states = mha_flash(query_states[0], key_states, value_states, "NHD", attn_mask)
        return hidden_states
