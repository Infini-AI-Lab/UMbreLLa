from __future__ import annotations
import torch
from transformers.integrations.fbgemm_fp8 import FbgemmFp8Linear

class FBGEMMFP8Linear:
    def __init__(self, dtype=torch.bfloat16):
        
        self.in_features = 0
        self.out_features = 0
        
        self.weight :torch.Tensor = None
        self.weight_scale :torch.Tensor = None
        self.input_scale_ub :torch.Tensor = None
        self.bias :torch.Tensor = None
        self.dtype = dtype
    
    def init_parameters(self, module: FbgemmFp8Linear):
        
        self.in_features = module.in_features
        self.out_features = module.out_features
       
        self.weight = module.weight.detach().pin_memory()
        self.weight_scale = module.weight_scale.detach().pin_memory()
        self.input_scale_ub = module.input_scale_ub.detach().pin_memory()
        if module.bias is not None:
            self.bias = module.bias.detach().pin_memory()
            self.bias = self.bias.to(self.dtype)
        else:
            self.bias = None
    
    def empty_like(self, module: FbgemmFp8Linear):
        
        self.in_features = module.in_features
        self.out_features = module.out_features
        
        
        self.weight =  module.weight.detach().clone()
        self.weight_scale = module.weight_scale.detach().clone()
        self.input_scale_ub = module.input_scale_ub.detach().clone()
        if module.bias is not None:
            self.bias = module.bias.detach().clone()
        
    def to(self, device, non_blocking=True):
        
        self.weight = self.weight.to(device, non_blocking=non_blocking)
        self.weight_scale = self.weight_scale.to(device, non_blocking=non_blocking)
        self.input_scale_ub = self.input_scale_ub.to(device, non_blocking=non_blocking)
        if self.bias is not None:
            self.bias = self.bias.to(device, non_blocking=non_blocking)
        
    
    def copy(self, module: FBGEMMFP8Linear, non_blocking=True):
        
        self.weight.copy_(module.weight, non_blocking=non_blocking)
        self.weight_scale.copy_(module.weight_scale, non_blocking=non_blocking)
        self.input_scale_ub.copy_(module.input_scale_ub, non_blocking=non_blocking)
        if self.bias is not None:
            self.bias.copy_(module.bias, non_blocking=non_blocking)
    
    
    def apply(self, x: torch.Tensor):
        
        
        num_tokens = None
        
        x_dtype = x.dtype
        x = x.to(torch.bfloat16)
        
        output_shape = (*x.shape[:-1], -1)
        
        x_quantized, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
            x.view(-1, x.shape[-1]), num_tokens, self.input_scale_ub
        )
        
        output = torch.ops.fbgemm.f8f8bf16_rowwise(
            x_quantized, self.weight, x_scale, self.weight_scale, use_fast_accum=True
        )
        
        
        output = output + self.bias if self.bias is not None else output
        
        output = output.to(x.device)
        output = output.reshape(output_shape)
        
        output = output.to(x_dtype)
        
        
        return output
        