from __future__ import annotations
import torch
from awq.modules.linear import WQLinear_GEMM
import awq_ext
class AwqLinear:
    def __init__(self):
        
        self.in_features = 0
        self.out_features = 0
        self.w_bit = 0
        self.group_size = 0
        self.qweight :torch.Tensor = None
        self.qzeros :torch.Tensor = None
        self.scales :torch.Tensor = None
        self.bias :torch.Tensor = None
        
    
    def init_parameters(self, module: WQLinear_GEMM):
        
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.w_bit = module.w_bit
        self.group_size = module.group_size
        self.qweight = module.qweight.detach().pin_memory()
        self.qzeros = module.qzeros.detach().pin_memory()
        self.scales = module.scales.detach().pin_memory()
        if module.bias is not None:
            self.bias = module.bias.detach()
        else:
            self.bias = None
    
    def empty_like(self, module: WQLinear_GEMM):
        
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.w_bit = module.w_bit
        self.group_size = module.group_size
        
        self.qweight = torch.zeros_like(module.qweight.detach())
        self.qzeros = torch.zeros_like(module.qzeros.detach())
        self.scales = torch.zeros_like(module.scales.detach())
        if module.bias is not None:
            self.bias = torch.zeros_like(module.bias.detach())
        
    def to(self, device, non_blocking=True):
        
        self.qweight = self.qweight.to(device, non_blocking=non_blocking)
        self.qzeros = self.qzeros.to(device, non_blocking=non_blocking)
        self.scales = self.scales.to(device, non_blocking=non_blocking)
        if self.bias is not None:
            self.bias = self.bias.to(device, non_blocking=non_blocking)
        
    
    def copy(self, module: AwqLinear, non_blocking=True):
        
        self.qweight.copy_(module.qweight, non_blocking=non_blocking)
        self.qzeros.copy_(module.qzeros, non_blocking=non_blocking)
        self.scales.copy_(module.scales, non_blocking=non_blocking)
        if self.bias is not None:
            self.bias.copy_(module.bias, non_blocking=non_blocking)
    
    
    def apply(self, x: torch.Tensor):
        
        out_shape = x.shape[:-1] + (self.out_features,)
      
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024

        if FP16_MATMUL_HEURISTIC_CONDITION:
                out = awq_ext.dequantize_weights_cuda(
                    self.qweight, self.scales, self.qzeros, 0, 0, 0, False
                )
                out = torch.matmul(x, out)
        else:
                out = awq_ext.gemm_forward_cuda(
                    x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8
                )
        
        
        out = out + self.bias if self.bias is not None else out
        out = out.reshape(out_shape)

        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out
        