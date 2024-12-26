import torch
from torch.nn.functional import softmax

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

def make_causal_mask_hf(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    _, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    return mask

def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = (p - q).relu_()
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1))
    return residual

def sampling_without_replacement(
        sampling_logits: torch.Tensor, 
        rand: torch.Tensor,  
        num_samples: int,
        temperature :float):

        sampling_q = softmax(sampling_logits / temperature, dim=-1)
        position = (rand.log()/sampling_q).topk(k=num_samples).indices.flatten()
        return position

def sampling_with_replacement(
        sampling_logits: torch.Tensor,   
        num_samples: int,
        temperature :float):

        #sampling_q = softmax(sampling_logits / temperature, dim=-1)
        sampling_q = softmax(sampling_logits / temperature, dim=-1)    
        position = sampling_q.multinomial(num_samples=num_samples, replacement=False).flatten()
        return position
def sampling_argmax(
        sampling_logits: torch.Tensor, 
        num_samples: int):
        return sampling_logits.topk(k=num_samples).indices.flatten()

def expand_kv(kv_cache, k):
    kv_shape = kv_cache[0][0].shape
    new_kv_cache = ()
    for kv in kv_cache:
        new_kv_cache = new_kv_cache + ([kv[0].expand(k, kv_shape[1], kv_shape[2], kv_shape[3]), 
                kv[1].expand(k, kv_shape[1], kv_shape[2], kv_shape[3])],)
    return new_kv_cache

def cat_kv(old_kv, delta_kv, cut_len :int):
    new_kv_cache = ()
    for i in range(len(old_kv)):
          k = torch.cat([old_kv[i][0], delta_kv[i][0][..., -cut_len:, :]], dim=-2)
          v = torch.cat([old_kv[i][1], delta_kv[i][1][..., -cut_len:, :]], dim=-2)
          new_kv_cache += ([k,v],)
    return new_kv_cache
    
    
def make_tree_attention_mask(
        prefix_len :int,
        gen_len :int,
        ancestors :list[list[int]],
        device ="cpu",
        dtype = torch.float32
    ) -> torch.FloatTensor:
    tree_mask = torch.full((gen_len, gen_len + prefix_len), torch.finfo(dtype).min, dtype=dtype).to(device=device)
    for idx, ancestor in enumerate(ancestors):
        if len(ancestor) > 0:
            tree_mask[idx][ancestor] = 0.0
    return tree_mask[None, None, :, :]


def get_sampling_logits(logits :torch.Tensor, top_p:float, T: float, replicate = False):
    if replicate:
        logits = logits.clone()
    if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                filter = cumulative_probs > top_p
                filter[..., 1:] = filter[..., :-1].clone()
                filter[..., 0] = 0
                indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                logits[indices_to_remove] = float('-inf')
    return logits

def select_kv(kv_cache: tuple[list[torch.FloatTensor]], indices: list[int]):
        new_kv_cache = ()
        for k,v in kv_cache:
             k = k[..., indices, :]
             v = v[..., indices, :]
             new_kv_cache += ([k,v],)
        return new_kv_cache


def cuda_graph_for_residual(device="cuda:0", dtype=torch.float16, dim=32000, n_warmups=3, mempool=None):
    static_p = torch.full((dim,), 1, dtype=dtype, device=device)
    static_q = torch.full((dim,), 0, dtype=dtype, device=device)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_residual = get_residual(
                    static_p,
                    static_q
                    )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
         static_residual = get_residual(
                    static_p,
                    static_q
                    )
    def run(p, q):
        static_p.copy_(p)
        static_q.copy_(q)
        graph.replay()
        return static_residual.clone()
    
    return run

def cuda_graph_for_sampling_without_replacement(
                device="cuda:0", dtype=torch.float16, 
                dim=32000, max_length=384, 
                n_warmups=3, mempool=None,
                idx_len = 8, num_samples = 16,
                temperature = 0.6, tree_size = 64):
    
    static_sampling_logits = torch.full((idx_len, dim), 1, dtype=dtype, device=device)
    static_rand = torch.empty((idx_len, dim), dtype=dtype, device=device).uniform_()

    

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_without_replacement(
                 static_sampling_logits,
                 static_rand,
                 num_samples,
                 temperature
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_without_replacement(
                 static_sampling_logits,
                 static_rand,
                 num_samples,
                 temperature
            )
    def run(draft_logits, rand_vector):
        static_sampling_logits.copy_(draft_logits)
        static_rand.copy_(rand_vector)
        graph.replay()
        return static_position.clone()
    
    return run

def cuda_graph_for_sampling_argmax(
                device="cuda:0", dtype=torch.float16, 
                dim=32000,
                n_warmups=3, mempool=None,
                idx_len = 8, num_samples = 16):
    
    static_sampling_logits = torch.full((idx_len, dim), 1, dtype=dtype, device=device)
    

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_argmax(
                 static_sampling_logits,
                 num_samples
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_argmax(
                 static_sampling_logits,
                 num_samples
            )
    def run(draft_logits):
        static_sampling_logits.copy_(draft_logits)
        graph.replay()
        return static_position.clone()
    
    return run


def cuda_graph_for_sampling_with_replacement(
                device="cuda:0", dtype=torch.float16, 
                dim=32000, max_length=384, 
                n_warmups=3, mempool=None,
                idx_len = 8, num_samples = 16,
                temperature = 0.6, tree_size = 64):
    
    static_sampling_logits = torch.full((idx_len, dim), 1, dtype=dtype, device=device)
    

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_with_replacement(
                 static_sampling_logits,
                 num_samples,
                 temperature
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_with_replacement(
                 static_sampling_logits,
                 num_samples,
                 temperature
            )
    def run(draft_logits):
        static_sampling_logits.copy_(draft_logits)
        graph.replay()
        return static_position.clone()
    
    return run

def contains_any_element(tensor, elements):
    """
    判断 tensor 是否包含 elements 列表中的至少一个整数元素。

    :param tensor: PyTorch Tensor
    :param elements: List of integers
    :return: True if at least one element is in tensor, False otherwise
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input 'tensor' must be a PyTorch Tensor.")

    if not all(isinstance(e, int) for e in elements):
        raise ValueError("All elements in 'elements' must be integers.")

    # 将 tensor 转换为 Python 集合，提高查找效率
    tensor_set = set(tensor.flatten().tolist())

    # 检查 elements 是否与 tensor_set 有交集
    return any(e in tensor_set for e in elements)


def find_first_element_position(tensor, elements):
    """
    查找列表中第一个出现在 tensor 中的元素的位置。

    :param tensor: PyTorch Tensor
    :param elements: List of integers
    :return: Index of the first matching element in the tensor, or -1 if none match
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input 'tensor' must be a PyTorch Tensor.")

    if not all(isinstance(e, int) for e in elements):
        raise ValueError("All elements in 'elements' must be integers.")

    # 展平 tensor 并转换为列表
    tensor_list = tensor.flatten().tolist()

    # 查找第一个匹配的元素
    for idx, value in enumerate(tensor_list):
        if value in elements:
            return idx

    return -1

def apply_repetition_penalty(input_ids: torch.LongTensor, logits: torch.FloatTensor, penalty: float):
        
        logit = torch.gather(logits, 1, input_ids)
        logit = torch.where(logit < 0, logit * penalty, logit / penalty)
        logits = logits.scatter(1, input_ids, logit)
        return logits

def apply_topk(logits: torch.FloatTensor, topk: int):

        top_k = min(topk, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        return logits

import re

def is_sentence_complete_regex(text):

    return bool(re.search(r'[.?!。？！]\s*$', text))