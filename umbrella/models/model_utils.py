import torch
import flashinfer

def make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    _, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(False, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), True)
    return mask


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def layer_norm(
    hidden_states: torch.Tensor,
    layernorm_variance_epsilon: float,
    layernorm_weight: torch.Tensor,
):  
    b, s, h = hidden_states.shape
    
    hidden_states = hidden_states.reshape(b * s, h)
    hidden_states = flashinfer.rmsnorm(hidden_states, layernorm_weight, layernorm_variance_epsilon)
    hidden_states = hidden_states.reshape(b, s, h)
    return hidden_states

def layer_norm_gemma(
    hidden_states: torch.Tensor,
    layernorm_variance_epsilon: float,
    layernorm_weight: torch.Tensor,
):  
    b, s, h = hidden_states.shape
    
    hidden_states = hidden_states.reshape(b * s, h)
    hidden_states = flashinfer.gemma_rmsnorm(hidden_states, layernorm_weight, layernorm_variance_epsilon)
    hidden_states = hidden_states.reshape(b, s, h)
    return hidden_states

def capture_graph(
    llm, decoding_seqlen :int =1, mempool=None, n_warmups :int=3
):
    device = llm.device
    bsz = llm.batch_size
    static_input_ids = torch.full((bsz, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_position_ids = torch.full((bsz, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_storage_ids = torch.arange(decoding_seqlen, dtype=torch.long, device=device)
    static_attn_mask = torch.full((decoding_seqlen, llm.max_length), 1, dtype=torch.bool, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = llm.inference(
                    input_ids=static_input_ids, 
                    position_ids=static_position_ids, 
                    attention_mask=static_attn_mask,
                    storage_ids=static_storage_ids, 
                    )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = llm.inference(
                input_ids=static_input_ids,  
                position_ids=static_position_ids, 
                attention_mask=static_attn_mask,
                storage_ids=static_storage_ids,
                )
    def run(input_ids, storage_ids, position_ids, attention_mask):
        static_input_ids.copy_(input_ids)
        static_storage_ids.copy_(storage_ids)
        static_position_ids.copy_(position_ids)
        static_attn_mask.copy_(attention_mask)
        graph.replay()
        return static_logits.clone()
    
    return run
