import torch
from ..models import AutoModelLM
from transformers import AutoTokenizer, GenerationConfig
from .speculation_utils import (
make_causal_mask, 
find_first_element_position, 
apply_repetition_penalty, 
apply_topk,
is_sentence_complete_regex
)
import time
import flashinfer
from ..logging_config import setup_logger
from ..utils import TextColors
from .base import BaseEngine
logger = setup_logger()

class AREngine(BaseEngine):
    def __init__(self,
        model_name: str,
        dtype=torch.float16,
        device :str = 'cuda:0',
        **kwargs
        ) -> None:

        super().__init__()

        self.model_name = model_name
        self.dtype = dtype
        self.device = device

        self.max_length = kwargs.pop("max_length", 8192)
        self.stop_distance = kwargs.pop("stop_distance", 32)
        self.safe_buffer = kwargs.pop("safe_buffer", 8)
        self.temperature = kwargs.pop("temperature", 0.0)
        self.topp = kwargs.pop("topp", 0.9)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.topk = kwargs.pop("topk", 32)
        self.num_beams = kwargs.pop("num_beams", 24)
        self.offload = kwargs.pop("offload", False)
        self.config = kwargs

    def initialize(self):

        self.tokens = torch.zeros(1, self.max_length, device=self.device).long()
        self.position_ids = torch.arange(self.max_length, device=self.device).unsqueeze(0)
        self.num_nodes = 0
        
        self.model = AutoModelLM.from_pretrained(
                    model_name=self.model_name, offload=self.offload, batch_size=1, 
                    max_length=self.max_length, device=self.device,
                    dtype=self.dtype)
        
        self.model.alloc(**self.config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.vocab_size = self.model.config.vocab_size
        self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.eos_tokens = self.generation_config.eos_token_id if (isinstance(self.generation_config.eos_token_id, list)) else [self.generation_config.eos_token_id]

    def prefill(self, text:str):
        input_ids = self.tokenizer.encode(text=text, return_tensors="pt").to(device=self.device)
        
        return self._prefill(input_ids=input_ids)
    
    def append(self, text:str):
        input_ids = self.tokenizer.encode(text=text, return_tensors="pt").to(device=self.device)
        input_ids = input_ids[:,1:]
        return self._append(input_ids)
    
    
    def _prefill(self, input_ids:torch.LongTensor):
        
        prefix_len = input_ids.shape[1]
        if prefix_len >= self.max_length - 2 * self.safe_buffer:
            return False
        self.num_nodes += prefix_len
        self.tokens[:,:prefix_len].copy_(input_ids)
        logits = self.model.inference(
             input_ids=self.tokens[:,:prefix_len],
             position_ids=self.position_ids[:,:prefix_len]
        )[0]

        
        next_token = logits[-1:].argmax(dim=-1, keepdim=True)
        
        self.tokens[:,self.num_nodes:self.num_nodes+1] = next_token

        return True
    
    def _append(self, input_ids:torch.LongTensor):
        append_len = input_ids.shape[1]
        if append_len + self.num_nodes >= self.max_length - 2 * self.safe_buffer:
            return False
        
        self.tokens[:,self.num_nodes+1:self.num_nodes+1+append_len].copy_(input_ids)
        num_last_iter_nodes = self.num_nodes
        self.num_nodes += (append_len + 1)
        
        logits = self.model.inference(
             input_ids=self.tokens[:,num_last_iter_nodes:self.num_nodes],
             position_ids=self.position_ids[:,num_last_iter_nodes:self.num_nodes],
        )[0]
        
        next_token = logits[-1:].argmax(dim=-1, keepdim=True)
        
        self.tokens[:,self.num_nodes:self.num_nodes+1] = next_token

        return True
    
    @torch.inference_mode()
    def decoding(self, max_new_tokens):
        
        decode = True
        generated_ids = []
        pos = 0
        start = self.num_nodes
        torch.cuda.synchronize()
        t1 = time.time()
        
        while decode and self.validate_status():
            begin_pos = self.num_nodes
            logits = self.model.inference(input_ids=self.tokens[:,self.num_nodes:self.num_nodes + 1],
            position_ids=self.position_ids[:,self.num_nodes:self.num_nodes + 1])[0]
            sampled_token = self.sample_tokens(logits)
            self.tokens[:,self.num_nodes + 1:self.num_nodes + 2].copy_(sampled_token)
            
            if sampled_token.item() in self.eos_tokens:
                decode = False
            
            self.num_nodes = self.num_nodes + 1
            generated_ids.extend(self.tokens[0,begin_pos:self.num_nodes].tolist())
            
            generated_text = (
                    self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
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
        logger.info(TextColors.colorize("Generated Tokens {:.2f} | TPOT {:.2f} ms ".format(dec_len, 1000 * (t2-t1)/dec_len), "magenta"))
        
        return dec_len, (t2 - t1)
    
    def validate_status(self):
 
        return self.num_nodes <= (self.max_length - self.safe_buffer)
    
    @torch.inference_mode()
    def reset(self):
        self.num_nodes = 0
        self.tokens.zero_()
        self.position_ids.zero_()
        self.model.clear()
    
    def update_generation_args(self, **generation_args):

        self.temperature = generation_args.pop("temperature", self.temperature)
        self.topp = generation_args.pop("topp", self.topp)
        self.repetition_penalty = generation_args.pop("repetition_penalty", self.repetition_penalty)
        self.topk = generation_args.pop("topk", self.topk)
        

    @torch.inference_mode()
    def sample_tokens(self, logits):
        # logits [bsz, seq, vocab]
        if self.temperature < 0.05:
            # greedy decoding
            sampled_tokens = logits.argmax(dim=-1)
        else:
            #stochastic decoding
            logits = apply_topk(logits, topk=self.topk)
            proba = torch.softmax(logits/self.temperature, dim=-1)
            proba = flashinfer.sampling.top_p_renorm_prob(proba, self.topp)
            sampled_tokens = torch.multinomial(proba, num_samples=1).squeeze(-1)
            
        return sampled_tokens
    
    
    def generate(self, **api_args):
        raise NotImplementedError
    

    def generate_stream(self, **api_args):
        raise NotImplementedError