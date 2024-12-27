import torch
from ..models import AutoModelLM
from transformers import AutoTokenizer, GenerationConfig
from .speculation_utils import (
make_causal_mask, 
find_first_element_position, 
apply_repetition_penalty, 
apply_topk,
is_sentence_complete_regex,
cuda_graph_for_sampling_argmax
)
import time
import flashinfer
import json
from ..logging_config import setup_logger
from ..utils import TextColors
from .base import BaseEngine
logger = setup_logger()

class StaticSpeculationEngine(BaseEngine):

    def __init__(self,
        draft_model_name: str,
        target_model_name: str,
        dtype=torch.float16,
        device :str = 'cuda:0',
        **kwargs
        ) -> None:
        
        
        self.draft_model_name = draft_model_name
        self.target_model_name = target_model_name
        self.dtype = dtype
        self.device = device
        self.growmap_path = kwargs.pop("growmap_path", None)
        assert self.growmap_path is not None, "Please specify growmap path for static trees"
        self.max_length = kwargs.pop("max_length", 8192)
        self.stop_distance = kwargs.pop("stop_distance", 32)
        self.safe_buffer = kwargs.pop("safe_buffer", 512)
        self.temperature = kwargs.pop("temperature", 0.0)
        self.topp = kwargs.pop("topp", 0.9)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.topk = kwargs.pop("topk", 32)
        self.config = kwargs

    
    def initialize(self):
        with open(self.growmap_path, 'r') as f:
            self.growmap = json.load(f)
        self.storage_ids = torch.arange(self.max_length).to(self.device)
        self.tokens = torch.zeros(1, self.max_length, device=self.device).long()
        
        self.Successors :list[list[int]] = []
        self.attn_mask = torch.full((self.max_length, 2 * self.max_length), False, dtype=torch.bool, device=self.device)
        self.position_ids = torch.zeros(1, self.max_length).long().to(self.device)
        self.attn_mask[:self.max_length, :self.max_length] = make_causal_mask((1, self.max_length),device=self.device)
        
        idx_lists = self.growmap["roots"]
        self.tree_depth = len(idx_lists)
        
        self.growmap_roots = []
        for x in idx_lists:
             self.growmap_roots.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.growmap["Successors"]
        tree_mask :torch.Tensor = torch.Tensor(self.growmap["mask"]).to(self.device)
        tree_mask = (tree_mask == 1)
        self.tree_mask = tree_mask
        self.node_in_path = self.tree_mask.int().sum(dim=-1)
        
        self.tree_size = self.growmap["size"]
        logger.info(TextColors.colorize("Tree Size {} | Tree Depth {}".format(self.tree_size - 1, self.tree_depth - 1), "magenta"))
        
        self.parents = torch.zeros(self.tree_size,dtype=torch.int32, device=self.device)
        for v, successor in enumerate(self.Successors):
            self.parents[successor] = v
        
        
        self.attn_mask[self.max_length - self.tree_size: self.max_length, self.max_length - self.tree_size: self.max_length] = tree_mask
        
        
        self.depth :torch.Tensor = torch.Tensor(self.growmap["depth"]).int().to(self.device)
        
        self.branch_lists = self.growmap['branches']
        graph_capture_list = [sum(x) for x in self.branch_lists if sum(x) > 0]
        graph_capture_list.append(1)
        
        self.draft_model = AutoModelLM.from_pretrained(
                    model_name=self.draft_model_name, offload=False, cuda_graph=True, batch_size=1, 
                    max_length=self.max_length, device=self.device,
                    dtype=self.dtype)
        
        self.draft_model.alloc(**self.config)
        
        self.target_model = AutoModelLM.from_pretrained(
                    model_name=self.target_model_name, offload=False, batch_size=1, 
                    max_length=self.max_length, device=self.device,
                    dtype=self.dtype)
        
        self.target_model.alloc(**self.config)
        
        self.draft_model.initialize_cuda_graph(graph_capture_list)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)
        self.vocab_size = self.target_model.config.vocab_size
        self.generation_config = GenerationConfig.from_pretrained(self.target_model_name)
        self.num_nodes = 0
        self.eos_tokens = self.generation_config.eos_token_id if (isinstance(self.generation_config.eos_token_id, list)) else [self.generation_config.eos_token_id]
        self.num_nodes = 0
        
        self.sampling_callables = {}
        self.sample_gather_indices = {}

        for i in range(self.tree_depth - 1):
            idx_len = len(idx_lists[i])
            num_samples = max(self.branch_lists[i])
            self.sampling_callables[i] = cuda_graph_for_sampling_argmax(device=self.device, 
            idx_len=idx_len, num_samples=num_samples, dtype=torch.float32, dim=self.vocab_size)
        
        for i in range(self.tree_depth - 1):
            ith_gather_list = []
            max_num_samples = max(self.branch_lists[i])
            for j, branch in enumerate(self.branch_lists[i]):
                branch_index = torch.arange(branch, device=self.device, dtype=torch.long)
                branch_index = branch_index + j * max_num_samples
                ith_gather_list.append(branch_index)
            ith_gather_list = torch.cat(ith_gather_list)
            self.sample_gather_indices[i] = ith_gather_list
        
    def prefill(self, text:str):
        input_ids = self.tokenizer.encode(text=text, return_tensors="pt").to(device=self.device)
        
        self._prefill(input_ids=input_ids)
    
    def append(self, text:str):
        input_ids = self.tokenizer.encode(text=text, return_tensors="pt").to(device=self.device)
        input_ids = input_ids[:,1:]
        self._append(input_ids)
    
    def _prefill(self, input_ids:torch.LongTensor):
        
        prefix_len = input_ids.shape[1]
        self.num_nodes += prefix_len
        self.num_nodes_this_iter = self.num_nodes + self.tree_size
        self.attn_mask_this_iter = self.attn_mask[self.max_length - self.num_nodes_this_iter: self.max_length, self.max_length - self.num_nodes_this_iter: 2 * self.max_length - self.num_nodes_this_iter].contiguous()
        self.num_draft_tokens_this_iter = self.num_nodes
        
        self.tokens[:,:prefix_len].copy_(input_ids)
        
        self.position_ids[:,:prefix_len] = torch.arange(prefix_len).unsqueeze(0)
        self.position_ids[:,prefix_len:prefix_len+self.tree_size] = prefix_len + self.depth

        self.draft_model.inference(
             input_ids=self.tokens[:,:prefix_len],
             storage_ids=self.storage_ids[:prefix_len],
             position_ids=self.position_ids[:,:prefix_len],
             attention_mask=self.attn_mask_this_iter[:prefix_len]
        )
    
        target_logits = self.target_model.inference(
             input_ids=self.tokens[:,:prefix_len],
             storage_ids=self.storage_ids[:prefix_len],
             position_ids=self.position_ids[:,:prefix_len],
             attention_mask=self.attn_mask_this_iter[:prefix_len]
        )[0]

        
        next_token = target_logits[-1:].argmax(dim=-1, keepdim=True)
        
        self.tokens[:,self.num_nodes:self.num_nodes+1] = next_token
    
    def _append(self, input_ids:torch.LongTensor):
        append_len = input_ids.shape[1]
        self.tokens[:,self.num_nodes+1:self.num_nodes+1+append_len].copy_(input_ids)
        num_last_iter_nodes = self.num_nodes
        self.num_nodes += (append_len + 1)
        self.num_nodes_this_iter = self.num_nodes + self.tree_size
        self.attn_mask_this_iter = self.attn_mask[self.max_length - self.num_nodes_this_iter: self.max_length, self.max_length - self.num_nodes_this_iter: 2 * self.max_length - self.num_nodes_this_iter].contiguous()
        self.num_draft_tokens_this_iter = self.num_nodes
        self.position_ids[:,:self.num_nodes] = torch.arange(self.num_nodes).unsqueeze(0)
        self.position_ids[:,self.num_nodes:self.num_nodes+self.tree_size] = self.num_nodes + self.depth
        
        self.draft_model.inference(
             input_ids=self.tokens[:,num_last_iter_nodes:self.num_nodes],
             storage_ids=self.storage_ids[num_last_iter_nodes:self.num_nodes],
             position_ids=self.position_ids[:,num_last_iter_nodes:self.num_nodes],
             attention_mask=self.attn_mask_this_iter[num_last_iter_nodes:self.num_nodes]
        )
    
        target_logits = self.target_model.inference(
             input_ids=self.tokens[:,num_last_iter_nodes:self.num_nodes],
             storage_ids=self.storage_ids[num_last_iter_nodes:self.num_nodes],
             position_ids=self.position_ids[:,num_last_iter_nodes:self.num_nodes],
             attention_mask=self.attn_mask_this_iter[num_last_iter_nodes:self.num_nodes]
        )[0]
        
        next_token = target_logits[-1:].argmax(dim=-1, keepdim=True)
        
        self.tokens[:,self.num_nodes:self.num_nodes+1] = next_token
    
    @torch.inference_mode()
    def speculative_decoding(self, max_new_tokens=128):
        max_new_tokens = max(max_new_tokens, self.stop_distance)
        torch.cuda.synchronize()
        t1 = time.time()
        large_model_step = 0
        decode = True
        start = self.num_nodes
        generated_ids = []
        pos = 0
        while decode:
            begin_pos = self.num_nodes
            self.build_tree()
            decode = self.verify()
            large_model_step = large_model_step + 1
            generated_ids.extend(self.tokens[0,begin_pos:self.num_nodes].tolist())
            
            generated_text = (
                    self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
                )

            now = len(generated_text) - 1
            
            if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now
            
            
            if (is_sentence_complete_regex(generated_text[-1]) and (self.num_nodes - start >= max_new_tokens - self.stop_distance)) or (self.num_nodes - start >= max_new_tokens):
                    decode = False
        
        print(" ".join(generated_text[pos:]), flush=True)
        torch.cuda.synchronize()
        t2 = time.time()
        dec_len = (self.num_nodes - start + 1)
        logger.info(TextColors.colorize("Avg Accept Tokens {:.2f} | TPOT {:.2f} ms ".format(dec_len/large_model_step, 1000 * (t2-t1)/dec_len), "magenta"))
        
        return dec_len, (t2 - t1), large_model_step
    
    @torch.inference_mode()
    def build_tree(self):
        
        for step in range(self.tree_depth):
            idx_list = self.growmap_roots[step]
            branch_list = self.branch_lists[step]
            total_branch = sum(branch_list)
            dec_len = len(idx_list)
            
            input_ids = self.tokens[:,self.num_draft_tokens_this_iter:self.num_draft_tokens_this_iter+dec_len]
            
            position_ids = self.position_ids[:,self.num_draft_tokens_this_iter:self.num_draft_tokens_this_iter+dec_len]
            storage_ids = self.storage_ids[self.num_draft_tokens_this_iter:self.num_draft_tokens_this_iter+dec_len]
            attention_mask = self.attn_mask_this_iter[self.num_draft_tokens_this_iter:self.num_draft_tokens_this_iter+dec_len]
            draft_logits = self.draft_model.graph_inference(
            input_ids=input_ids,
            storage_ids=storage_ids,
            position_ids=position_ids,
            attention_mask=attention_mask
            )[0]
    
            self.num_draft_tokens_this_iter += dec_len
            if step < self.tree_depth - 1:
                new_tokens_set = self.sampling_callables[step](draft_logits)
                self.tokens[0,self.num_draft_tokens_this_iter: self.num_draft_tokens_this_iter + total_branch] = new_tokens_set[self.sample_gather_indices[step]]
    
    @torch.inference_mode()
    def verify(self):
        
        input_ids = self.tokens[:,self.num_nodes:self.num_draft_tokens_this_iter]
        position_ids = self.position_ids[:,self.num_nodes:self.num_draft_tokens_this_iter]
        storage_ids = self.storage_ids[self.num_nodes:self.num_draft_tokens_this_iter]
        attention_mask = self.attn_mask_this_iter[self.num_nodes:self.num_draft_tokens_this_iter]
        
        target_logits = self.target_model.inference(
             input_ids=input_ids,
             storage_ids=storage_ids,
             position_ids=position_ids,
             attention_mask=attention_mask
        )[0]
        
        
        if self.repetition_penalty > 1.01:
            target_logits = apply_repetition_penalty(
                self.tokens[:,:self.num_nodes + 1].expand(self.num_draft_tokens_this_iter - self.num_nodes, -1),
                target_logits,
                self.repetition_penalty 
            )
        
        if self.temperature < 0.05:
            # greedy decoding
            sampled_tokens = target_logits.argmax(dim=-1)
        
        else:
            #stochastic decoding
            proba = torch.softmax(target_logits/self.temperature, dim=-1)
            proba = flashinfer.sampling.top_k_renorm_prob(proba, self.topk)
            proba = flashinfer.sampling.top_p_renorm_prob(proba, self.topp)
            sampled_tokens = torch.multinomial(proba, num_samples=1).squeeze(-1)
            
        speculated_tokens = self.tokens[0, self.num_nodes:self.num_draft_tokens_this_iter]
        ref_tokens = sampled_tokens[self.parents]
        accept = (ref_tokens == speculated_tokens)
        accept[0] = True
        accept = accept[None,:].repeat(self.tree_size, 1)
        
        accept_node_in_path = (accept * self.tree_mask).int().sum(dim=-1)
        accept_path = (accept_node_in_path == self.node_in_path).nonzero().squeeze_(-1)
        
        
        target_token = sampled_tokens[accept_path[-1]]
        accept_length = accept_path.shape[0]
        accept_tokens = speculated_tokens[accept_path]
        self.tokens[0, self.num_nodes:self.num_nodes + accept_length] = accept_tokens
        self.tokens[0, self.num_nodes + accept_length] = target_token
        
        continue_generation = True
        eos_position = find_first_element_position(self.tokens[0, self.num_nodes:self.num_nodes + accept_length + 1], self.eos_tokens)
        if eos_position >= 0:
            continue_generation = False
            accept_path = accept_path[:eos_position]
            accept_length = len(accept_path)
        
        accept_path += self.num_nodes
        self.draft_model.kv_cache.gather_kv_incremental(accept_path, self.num_nodes)
        self.target_model.kv_cache.gather_kv_incremental(accept_path, self.num_nodes)
        
        num_last_iter_nodes = self.num_nodes
        self.num_nodes = self.num_nodes + accept_length
        self.num_draft_tokens_this_iter = self.num_nodes
        
        self.num_nodes_this_iter = self.num_nodes + self.tree_size
        self.attn_mask_this_iter = self.attn_mask[self.max_length - self.num_nodes_this_iter: 2 * self.max_length - self.num_nodes_this_iter, self.max_length - self.num_nodes_this_iter: 2 * self.max_length - self.num_nodes_this_iter].contiguous()
        
        if accept_length > 0:
            self.position_ids[:,num_last_iter_nodes:self.num_nodes] = self.position_ids[:,accept_path]
        self.position_ids[:,self.num_nodes:self.num_nodes+self.tree_size] = self.num_nodes + self.depth
        
        return continue_generation
    
    @torch.inference_mode()
    def reset(self):
        self.num_nodes = 0
        self.tokens.zero_()
        self.position_ids.zero_()
        self.draft_model.clear()
        self.target_model.clear()
    
    
    def validate_status(self):
        
        return self.num_nodes <= (self.max_length - self.safe_buffer)
    
    def update_generation_args(self, **generation_args):
        
        self.temperature = generation_args.pop("temperature", self.temperature)
        self.topp = generation_args.pop("topp", self.topp)
        self.repetition_penalty = generation_args.pop("repetition_penalty", self.repetition_penalty)
        self.topk = generation_args.pop("topk", self.topk)
        