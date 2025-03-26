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
        self.weight_scale = self.weight_scale.to(self.dtype)
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
        
        x_dtype = x.dtype 
        x = x.to(torch.bfloat16)

        output_shape = (*x.shape[:-1], -1)

        x_dequantized = x 
        weight_dequantized = self.weight.to(torch.bfloat16) * self.weight_scale  # 反量化权重

        output = torch.matmul(x_dequantized, weight_dequantized.T)

        
        if self.bias is not None:
            output = output + self.bias.to(torch.bfloat16)

        output = output.to(x_dtype)
        output = output.reshape(output_shape)

        return output
            