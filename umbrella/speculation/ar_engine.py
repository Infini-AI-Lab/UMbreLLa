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

class AREngine:

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
        pass

    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        past_len = self.model.kv_cache.get_kv_len()
        position_ids = torch.arange(past_len, past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        return position_ids

    @torch.inference_mode()
    def _prefill(self, input_ids:torch.LongTensor):
        
        prefix_len = input_ids.shape[1]
        if prefix_len >= self.max_length - 2 * self.safe_buffer:
            return False
                
        self.tokens[:,:prefix_len].copy_(input_ids)

        logits = self.model.inference(input_ids=self.tokens[:,:prefix_len], position_ids=self.get_ctx(input_ids))
        assert self.model.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.model.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"
        
        next_token = self.sample_tokens(logits[:, -1, :])
        
        self.tokens[:,prefix_len:prefix_len+1] = next_token

        return True
    
    @torch.inference_mode()
    def _append(self, input_ids:torch.LongTensor):
        pass
    
    
    def update_generation_args(self, **generation_args):
        
        self.temperature = generation_args.pop("temperature", self.temperature)
        self.topp = generation_args.pop("topp", self.topp)
        self.repetition_penalty = generation_args.pop("repetition_penalty", self.repetition_penalty)
        self.topk = generation_args.pop("topk", self.topk)
        
    @torch.inference_mode()
    def reset(self):
        self.tokens.zero_()
        self.model.clear()
    
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

    def validate_status(self):
        
        return self.model.kv_cache.get_kv_len() < self.max_length - self.safe_buffer

    @torch.inference_mode()
    def generate(self, **api_args):
        
        self.update_generation_args(**api_args)
        input_ids = api_args.get("input_ids", None)
        max_new_tokens = api_args.get("max_new_tokens", 128)
        
        if input_ids is None:
            context = api_args.get("context", None)
            if context is None or len(context) == 0 or max_new_tokens == 0:
                api_args["generated_text"] = ""
                api_args["generated_tokens"] = []
                api_args["avg_accept_tokens"] = 0
                api_args["time_per_output_token"] = 0
                return api_args
            success = self.prefill(context)
        
        else:
            if len(input_ids) == 0 or max_new_tokens == 0:
                api_args["generated_text"] = ""
                api_args["generated_tokens"] = []
                api_args["avg_accept_tokens"] = 0
                api_args["time_per_output_token"] = 0
                return api_args
            input_ids = torch.Tensor(input_ids).long().unsqueeze(0).to(self.device)
            success = self._prefill(input_ids=input_ids)
        
        if not success:
            api_args["generated_text"] = ""
            api_args["generated_tokens"] = []
            api_args["avg_accept_tokens"] = 0
            api_args["time_per_output_token"] = 0
            self.reset()
            return api_args
    
        start = self.model.kv_cache.get_kv_len()
        next_token = self.tokens[:, self.model.kv_cache.get_kv_len()-1:self.model.kv_cache.get_kv_len()]
        torch.cuda.synchronize()
        t1 = time.time()
        
        while (self.model.kv_cache.get_kv_len() - start) < max_new_tokens and self.validate_status():
            logits = self.model.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            next_token = self.sample_tokens(logits[:, -1, :])
            self.tokens[:,self.model.kv_cache.get_kv_len():self.model.kv_cache.get_kv_len()+1] = next_token
            if next_token in self.eos_tokens:
                break
        
        torch.cuda.synchronize()
        t2 = time.time()
        
        dec_len = (self.model.kv_cache.get_kv_len() - start + 1)
        generated_text = self.tokenizer.decode(
            self.tokens[0,start:self.model.kv_cache.get_kv_len()+1].tolist(), 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        api_args["generated_text"] = generated_text
        api_args["generated_tokens"] = self.tokens[0,start:self.model.kv_cache.get_kv_len()+1].tolist()
        api_args["time_per_output_token"] = 1000 * (t2-t1)/dec_len
        self.reset()
        return api_args
        
        
    @torch.inference_mode()
    def generate_stream(self, **api_args):
        """
        Gradio
        """
        self.update_generation_args(**api_args)
        input_ids = api_args.get("input_ids", None)
        max_new_tokens = api_args.get("max_new_tokens", 128)
        
        if input_ids is None:
            context = api_args.get("context", None)
            if context is None or len(context) == 0 or max_new_tokens == 0:
                api_args["generated_text"] = ""
                api_args["generated_tokens"] = []
                api_args["avg_accept_tokens"] = 0
                api_args["time_per_output_token"] = 0
                return api_args
            success = self.prefill(context)
        
        else:
            if len(input_ids) == 0 or max_new_tokens == 0:
                api_args["generated_text"] = ""
                api_args["generated_tokens"] = []
                api_args["avg_accept_tokens"] = 0
                api_args["time_per_output_token"] = 0
                return api_args
            input_ids = torch.Tensor(input_ids).long().unsqueeze(0).to(self.device)
            success = self._prefill(input_ids=input_ids)
        
        if not success:
            yield "Exceeding reserved allowed context length", "Exceeding reserved allowed context length"

        torch.cuda.synchronize()
        t1 = time.time()
        start = self.model.kv_cache.get_kv_len()
        generated_ids = []
        pos = 0

        partial_text = ""

        while self.validate_status():
            begin_pos = self.model.kv_cache.get_kv_len()

            logits = self.model.inference(input_ids=self.tokens[:, begin_pos - 1 : begin_pos], position_ids=self.get_ctx(self.tokens[:, begin_pos - 1 : begin_pos]))
            next_token = self.sample_tokens(logits[:, -1, :])
            self.tokens[:, begin_pos : begin_pos + 1] = next_token

            new_ids = self.tokens[0, begin_pos : begin_pos + 1].tolist()
            generated_ids.extend(new_ids)

            generated_text_list = (
                self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
            )

            now = len(generated_text_list) - 1

            if now > pos:
                new_text_chunk = " ".join(generated_text_list[pos:now]) + " "
                partial_text += new_text_chunk
                
                t2 = time.time()
                dec_len = (self.model.kv_cache.get_kv_len() - start + 1)
                
                perf_log = "Output Tokens {} | TPOT {:.2f} ms ".format(
                    dec_len, 1000 * (t2 - t1) / dec_len
                )
                
                yield partial_text, perf_log

                pos = now

            if (
                is_sentence_complete_regex(generated_text_list[-1]) 
                and (self.model.kv_cache.get_kv_len() - start >= max_new_tokens)
            ) or ((self.model.kv_cache.get_kv_len() - start) >= max_new_tokens):
                decode = False


        final_piece = " ".join(generated_text_list[pos:])
        if final_piece:
            partial_text += final_piece

        t2 = time.time()
        dec_len = (self.model.kv_cache.get_kv_len() - start + 1)
                
        perf_log = "Output Tokens {} | Avg Accept Tokens {:.2f} | TPOT {:.2f} ms ".format(
                    dec_len, 1000 * (t2 - t1) / dec_len
                )
        yield partial_text, perf_log

        torch.cuda.synchronize()
        t2 = time.time()
        dec_len = (self.model.kv_cache.get_kv_len() - start + 1)
        logger.info(
            TextColors.colorize(
                "Output Tokens {} | Avg Accept Tokens {:.2f} | TPOT {:.2f} ms ".format(
                    dec_len, 1000 * (t2 - t1) / dec_len
                ),
                "magenta",
            )
        )

        
        self.reset()
        
