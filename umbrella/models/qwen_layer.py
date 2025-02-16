from __future__ import annotations
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from ..quantization.awq_utils import AwqLinear
class QwenLayer:
    def __init__(self, layer_idx, device = "cpu") -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.bq :torch.Tensor = None
        self.bk :torch.Tensor = None
        self.bv :torch.Tensor = None
        
        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx
        self.device = device
    def init_parameters(self, hf_layer: Qwen2DecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()

        self.bq :torch.Tensor= hf_layer.self_attn.q_proj.bias.detach()
        self.bk :torch.Tensor= hf_layer.self_attn.k_proj.bias.detach()
        self.bv :torch.Tensor= hf_layer.self_attn.v_proj.bias.detach()
        
        
        
        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    
    def to(self, device:str = 'cuda:0', non_blocking = True):

        self.device = device
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=non_blocking)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=non_blocking)
        self.wq = self.wq.to(device, non_blocking=non_blocking)
        self.wk = self.wk.to(device, non_blocking=non_blocking)
        self.wv = self.wv.to(device, non_blocking=non_blocking)
        self.bq = self.bq.to(device, non_blocking=non_blocking)
        self.bk = self.bk.to(device, non_blocking=non_blocking)
        self.bv = self.bv.to(device, non_blocking=non_blocking)
        self.wo = self.wo.to(device, non_blocking=non_blocking)
        self.gate_proj = self.gate_proj.to(device, non_blocking=non_blocking)
        self.up_proj = self.up_proj.to(device, non_blocking=non_blocking)
        self.down_proj =  self.down_proj.to(device, non_blocking=non_blocking)

    def copy(self, layer: QwenLayer):

        self.wq.copy_(layer.wq, non_blocking=True)
        self.wk.copy_(layer.wk, non_blocking=True)
        self.wv.copy_(layer.wv, non_blocking=True)
        
        self.bq.copy_(layer.bq, non_blocking=True)
        self.bk.copy_(layer.bk, non_blocking=True)
        self.bv.copy_(layer.bv, non_blocking=True)
        
        self.wo.copy_(layer.wo, non_blocking=True)
        self.gate_proj.copy_(layer.gate_proj, non_blocking=True)
        self.up_proj.copy_(layer.up_proj, non_blocking=True)
        self.down_proj.copy_(layer.down_proj, non_blocking=True)
        
        self.input_layernorm_weight.copy_(layer.input_layernorm_weight, non_blocking=True)
        self.post_attention_layernorm_weight.copy_(layer.post_attention_layernorm_weight, non_blocking=True)
        self.input_layernorm_variance_epsilon= layer.input_layernorm_variance_epsilon
        self.post_attention_layernorm_variance_epsilon = layer.post_attention_layernorm_variance_epsilon
        self.layer_idx = layer.layer_idx
        
    def alloc_space(self, layer: QwenLayer, device):

        self.device = device
        self.wq = torch.zeros_like(layer.wq).to(device)
        self.wk = torch.zeros_like(layer.wk).to(device)
        self.wv = torch.zeros_like(layer.wv).to(device)
        self.bq = torch.zeros_like(layer.bq).to(device)
        self.bk = torch.zeros_like(layer.bk).to(device)
        self.bv = torch.zeros_like(layer.bv).to(device)
        
        
        self.wo = torch.zeros_like(layer.wo).to(device)


        self.gate_proj = torch.zeros_like(layer.gate_proj).to(device)
        self.up_proj = torch.zeros_like(layer.up_proj).to(device)
        self.down_proj = torch.zeros_like(layer.down_proj).to(device)
        self.input_layernorm_weight = torch.zeros_like(layer.input_layernorm_weight).to(device)
        self.post_attention_layernorm_weight = torch.zeros_like(layer.post_attention_layernorm_weight).to(device)


class QwenAwqLayer():
    def __init__(self, layer_idx, device="cpu"):
        
        self.wq = AwqLinear()
        self.wk = AwqLinear()
        self.wv = AwqLinear()
        self.wo = AwqLinear()

        self.gate_proj = AwqLinear()
        self.up_proj = AwqLinear()
        self.down_proj = AwqLinear()
        
        
        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx
        self.device = device
        
    
    def init_parameters(self, hf_layer):
        
        
        self.wq.init_parameters(hf_layer.self_attn.q_proj)
        self.wk.init_parameters(hf_layer.self_attn.k_proj)
        self.wv.init_parameters(hf_layer.self_attn.v_proj)
        self.wo.init_parameters(hf_layer.self_attn.o_proj)
        self.gate_proj.init_parameters(hf_layer.mlp.gate_proj)
        self.up_proj.init_parameters(hf_layer.mlp.up_proj)
        self.down_proj.init_parameters(hf_layer.mlp.down_proj)
        
        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach().pin_memory()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach().pin_memory()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon
    
    def to(self, device:str = 'cuda:0', non_blocking = True):
        
        self.device = device
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=non_blocking)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=non_blocking)
        
        self.wq.to(device=device)
        self.wk.to(device=device)
        self.wv.to(device=device)
        self.wo.to(device=device)
        
        self.gate_proj.to(device=device)
        self.up_proj.to(device=device)
        self.down_proj.to(device=device)
        
    def alloc_space(self, layer: QwenAwqLayer, device):

        self.device = device
        self.wq.empty_like(layer.wq)
        self.wk.empty_like(layer.wk)
        self.wv.empty_like(layer.wv)
        self.wo.empty_like(layer.wo)

        self.gate_proj.empty_like(layer.gate_proj)
        self.up_proj.empty_like(layer.up_proj)
        self.down_proj.empty_like(layer.down_proj)
        
        self.wq.to(device=device)
        self.wk.to(device=device)
        self.wv.to(device=device)
        self.wo.to(device=device)
        
        self.gate_proj.to(device=device)
        self.up_proj.to(device=device)
        self.down_proj.to(device=device)
        
        self.input_layernorm_weight = torch.zeros_like(layer.input_layernorm_weight).to(device)
        self.post_attention_layernorm_weight = torch.zeros_like(layer.post_attention_layernorm_weight).to(device)
    
    def copy(self, layer: QwenAwqLayer):

        self.wq.copy(layer.wq, non_blocking=True)
        self.wk.copy(layer.wk, non_blocking=True)
        self.wv.copy(layer.wv, non_blocking=True)
        self.wo.copy(layer.wo, non_blocking=True)
        self.gate_proj.copy(layer.gate_proj, non_blocking=True)
        self.up_proj.copy(layer.up_proj, non_blocking=True)
        self.down_proj.copy(layer.down_proj, non_blocking=True)
        
        self.input_layernorm_weight.copy_(layer.input_layernorm_weight, non_blocking=True)
        self.post_attention_layernorm_weight.copy_(layer.post_attention_layernorm_weight, non_blocking=True)
        self.input_layernorm_variance_epsilon= layer.input_layernorm_variance_epsilon
        self.post_attention_layernorm_variance_epsilon = layer.post_attention_layernorm_variance_epsilon
        self.layer_idx = layer.layer_idx


class QwenPackedLayer:
    def __init__(self, layer_idx, device = "cpu") -> None:
        
        self.wqkv :torch.Tensor = None
        self.bqkv :torch.Tensor = None
        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx
        self.device = device
    def init_parameters(self, hf_layer: Qwen2DecoderLayer):

        self.wqkv :torch.Tensor= torch.cat(
        [
        hf_layer.self_attn.q_proj.weight.detach(),
        hf_layer.self_attn.k_proj.weight.detach(),
        hf_layer.self_attn.v_proj.weight.detach(),
        ],
        dim=0
        )
        
        self.bqkv :torch.Tensor= torch.cat(
        [
        hf_layer.self_attn.q_proj.bias.detach(),
        hf_layer.self_attn.k_proj.bias.detach(),
        hf_layer.self_attn.v_proj.bias.detach(),
        ],
        dim=0
        )
        
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()
        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    
    def to(self, device:str = 'cuda:0', non_blocking = True):

        self.device = device
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=non_blocking)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=non_blocking)
        self.wqkv = self.wqkv.to(device, non_blocking=non_blocking)
        self.bqkv = self.bqkv.to(device, non_blocking=non_blocking)
        self.wo = self.wo.to(device, non_blocking=non_blocking)
        self.gate_proj = self.gate_proj.to(device, non_blocking=non_blocking)
        self.up_proj = self.up_proj.to(device, non_blocking=non_blocking)
        self.down_proj =  self.down_proj.to(device, non_blocking=non_blocking)

    def copy(self, layer: QwenPackedLayer):

        self.wqkv.copy_(layer.wqkv, non_blocking=True)
        self.bqkv.copy_(layer.bqkv, non_blocking=True)
        self.wo.copy_(layer.wo, non_blocking=True)
        self.gate_proj.copy_(layer.gate_proj, non_blocking=True)
        self.up_proj.copy_(layer.up_proj, non_blocking=True)
        self.down_proj.copy_(layer.down_proj, non_blocking=True)
        
        self.input_layernorm_weight.copy_(layer.input_layernorm_weight, non_blocking=True)
        self.post_attention_layernorm_weight.copy_(layer.post_attention_layernorm_weight, non_blocking=True)
        self.input_layernorm_variance_epsilon= layer.input_layernorm_variance_epsilon
        self.post_attention_layernorm_variance_epsilon = layer.post_attention_layernorm_variance_epsilon
        self.layer_idx = layer.layer_idx
        
    def alloc_space(self, layer: QwenPackedLayer, device):

        self.device = device
        self.wqkv = torch.zeros_like(layer.wqkv).to(device)
        self.bqkv = torch.zeros_like(layer.bqkv).to(device)
        self.wo = torch.zeros_like(layer.wo).to(device)


        self.gate_proj = torch.zeros_like(layer.gate_proj).to(device)
        self.up_proj = torch.zeros_like(layer.up_proj).to(device)
        self.down_proj = torch.zeros_like(layer.down_proj).to(device)
        self.input_layernorm_weight = torch.zeros_like(layer.input_layernorm_weight).to(device)
        self.post_attention_layernorm_weight = torch.zeros_like(layer.post_attention_layernorm_weight).to(device)