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
# Some code comes from MInference
# Original license:
# Copyright (c) Microsoft Corporation. and affiliates All rights reserved.
#
# See LICENSE.txt for license information
################################################################################

import torch
from torch.nn import functional as F

from flashinfer.norm import rmsnorm
from minference import vertical_slash_sparse_attention, block_sparse_attention, streaming_forward

#fix the kernel part later
from kernels import shadowkv

def layer_norm(
    hidden_states: torch.Tensor,
    eps: float,
    w: torch.Tensor,
):
    return rmsnorm(hidden_states.view(-1, hidden_states.size(-1)), w, eps).view_as(hidden_states)

# def layer_norm(
#     hidden_states: torch.Tensor,
#     eps: float,
#     w: torch.Tensor,
# ):
#     input_dtype = hidden_states.dtype
#     hidden_states = hidden_states.to(torch.float32)
#     variance = hidden_states.pow(2).mean(-1, keepdim=True)
#     hidden_states = hidden_states * torch.rsqrt(variance + eps)
#     hidden_states = w * hidden_states.to(input_dtype)
#     return hidden_states


# copy from https://github.com/microsoft/MInference/blob/main/minference/modules/minference_forward.py

last_q = 64
arange = torch.arange(last_q, device="cuda")
LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]

def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device) # Zero matrix used for padding
    mat_padded =  torch.cat((zero_mat, mat, zero_mat), -1) # pads the matrix on left and right
    mat_strided = mat_padded.as_strided((1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1)) # Change the strides
    sum_diags = torch.sum(mat_strided, 2) # Sums the resulting matrix's columns
    return sum_diags[:,:,1:]

def minference_prefill_kernel(query_states, key_states, value_states, minference_parttern):
    q_len = query_states.shape[2] # [bsz, heads, q_len, head_dim]
    bsz = query_states.shape[0]
    gqa_groups = query_states.shape[1] // key_states.shape[1]

    assert q_len > 1
    output = torch.empty_like(query_states)
    for head in range(query_states.size(1)):
        q = query_states[:, head, :, :].unsqueeze(1)
        k = key_states[:, head // gqa_groups, :, :].unsqueeze(1)
        v = value_states[:, head // gqa_groups, :, :].unsqueeze(1)
        attn_output = gather_last_q_vertical_slash_topk_v4(q, k, v, head, minference_parttern)
        output[:, head:head + 1] = attn_output

    return output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)


def gather_last_q_vertical_slash_topk_v4(q, k, v, head_id, minference_parttern):
    def vertical_and_slash_kernel(q, k, v, vertical_size, slash_size):
        vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
        last_q = min(64, q_len)
        qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q:,:], k)
        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:].to(q.device), qk[:, :, :, -last_q:], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[...,-100:] = torch.inf
        slash_topk = slash
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)

    def block_sparse_kernel(q, k, v, vertical_size=None, slash_size=None):
        topk = 100
        return block_sparse_attention(q, k, v, topk)

    q_len = q.shape[2]
    bsz = q.shape[0]

    ty, vertical_size, slash_size, _ = minference_parttern.get(head_id)

    fc = {
        "stream_llm": streaming_forward,
        "vertical_and_slash": vertical_and_slash_kernel,
        "block_sparse": block_sparse_kernel,
    }[ty]
    return fc(q, k, v, vertical_size, slash_size)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    # if position_ids shape is (batch_size, num_heads, seq_len), then reshape it to (batch_size*num_heads, seq_len)
    if len(position_ids.shape) == 3:
        position_ids = position_ids.view(-1, position_ids.size(-1))
        cos = cos[position_ids]
        sin = sin[position_ids]
        q_embed = (q * cos) + (rotate_half(q) * sin)

    else:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


def apply_rotary_pos_emb_cuda(x, cos_sin, position_ids):
    batch_size, heads, seq_len, embed_dim = x.shape
    half_dim = embed_dim // 2
    
    output = torch.empty_like(x)

    shadowkv.apply_rotary_pos_emb_new(
        x, cos_sin, position_ids, output,
        int(batch_size), int(heads), int(seq_len), int(embed_dim),
        int(x.stride(0)), int(x.stride(1)), int(x.stride(2)), int(x.stride(3)),
        int(cos_sin.stride(0)),
        int(position_ids.stride(0)), int(position_ids.stride(1)), int(position_ids.stride(2)),
        int(half_dim)
    )
    
    return output

def apply_rotary_pos_emb_cuda_push_cache(x, cos_sin, position_ids, chunk_size, cache, sparse_start, sparse_end, cnts):
    batch_size, heads, seq_len, embed_dim = x.shape
    half_dim = embed_dim // 2
    if cos_sin.shape[-1] == 128:
        shadowkv.apply_rotary_pos_emb_push_cache_opt(
            x, cos_sin, position_ids, cache, cnts,
            int(batch_size), int(heads), int(seq_len), int(embed_dim),
            int(x.stride(0)), int(x.stride(1)), int(x.stride(2)), int(x.stride(3)),
            int(cos_sin.stride(0)),
            int(position_ids.stride(0)), int(position_ids.stride(1)), int(position_ids.stride(2)),
            int(cache.stride(0)), int(cache.stride(1)), int(cache.stride(2)),
            int(sparse_start), int(sparse_end),
            int(half_dim), int(chunk_size)
        )
    elif cos_sin.shape[-1] == 64:
        shadowkv.apply_rotary_pos_emb_push_cache_opt_glm(
            x, cos_sin, position_ids, cache, cnts,
            int(batch_size), int(heads), int(seq_len), int(embed_dim),
            int(x.stride(0)), int(x.stride(1)), int(x.stride(2)), int(x.stride(3)),
            int(cos_sin.stride(0)),
            int(position_ids.stride(0)), int(position_ids.stride(1)), int(position_ids.stride(2)),
            int(cache.stride(0)), int(cache.stride(1)), int(cache.stride(2)),
            int(sparse_start), int(sparse_end),
            int(half_dim), int(chunk_size)
        )
    else:
        raise ValueError(f"Invalid cos_sin shape {cos_sin.shape}")
    
    return cache

def batch_gather_gemm_rotary_pos_emb_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
    output: torch.Tensor,
    chunk_size: int,
    cache: torch.Tensor,
    sparse_start: int,
    sparse_end: int,
    cnts: torch.Tensor
):
    batch_size, seq_len, rank = a.shape
    _, heads, head_dim, _ = b.shape
    max_seq_len, _ = cos_sin.shape
    _, _, num_chunks = position_ids.shape
    sparse_budget = num_chunks * chunk_size
    position_ids = position_ids.to(torch.int32).contiguous()
    
    shadowkv.batch_gather_gemm(
        a.contiguous(),
        b.contiguous(),
        cos_sin.contiguous(),
        cos_sin.contiguous(),
        position_ids,
        output,
        batch_size,
        heads,
        seq_len,
        head_dim,
        rank,
        sparse_budget,
        max_seq_len,
        chunk_size,
        cnts,
    )

    return apply_rotary_pos_emb_cuda_push_cache(output, cos_sin, position_ids, chunk_size, cache, sparse_start, sparse_end, cnts)


# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

def norm_logits(logits : torch.Tensor, temperature=0.6, top_k=-1, top_p=0.9) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)
    return probs


def sample(probs : torch.Tensor, num_samples=1):
    idx_next = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    return idx_next

def sample_token(logits: torch.Tensor, temperature=0, top_k=50, top_p=0.9):
    if temperature == 0.0:
        token = logits.argmax(dim=-1, keepdim=True)
    else:
        token = sample(norm_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k))
    
    return token