from __future__ import annotations
import torch
from transformers.models.glm.modeling_glm import GlmDecoderLayer
from ..quantization.awq_utils import AwqLinear


# refers to https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm/modeling_glm.py#L319

class GLM4Layer:
    def __init__(self, layer_idx, device="cpu") -> None:
        self.wq: torch.Tensor = None
        self.wk: torch.Tensor = None
        self.wv: torch.Tensor = None
        self.wo: torch.Tensor = None

        self.gate_up_proj: torch.Tensor = None
        self.down_proj: torch.Tensor = None

        self.input_layernorm_weight: torch.Tensor = None
        self.input_layernorm_variance_epsilon: float = 0.0

        self.post_attention_layernorm_weight: torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon: float = 0.0

        self.layer_idx = layer_idx
        self.device = device

    def init_parameters(self, hf_layer: GlmDecoderLayer):
        self.wq: torch.Tensor = hf_layer.self_attn.q_proj.weight.detach()
        self.wk: torch.Tensor = hf_layer.self_attn.k_proj.weight.detach()
        self.wv: torch.Tensor = hf_layer.self_attn.v_proj.weight.detach()
        self.wo: torch.Tensor = hf_layer.self_attn.o_proj.weight.detach()

        self.gate_up_proj = hf_layer.mlp.gate_up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    def to(self, device: str = 'cuda:0', non_blocking=True):
        self.device = device
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=non_blocking)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device,
                                                                                       non_blocking=non_blocking)

        self.wq = self.wq.to(device, non_blocking=non_blocking)
        self.wk = self.wk.to(device, non_blocking=non_blocking)
        self.wv = self.wv.to(device, non_blocking=non_blocking)
        self.wo = self.wo.to(device, non_blocking=non_blocking)
        self.gate_up_proj = self.gate_up_proj.to(device, non_blocking=non_blocking)
        self.down_proj = self.down_proj.to(device, non_blocking=non_blocking)

    def copy(self, layer: GLM4Layer):
        self.wq.copy_(layer.wq, non_blocking=True)
        self.wk.copy_(layer.wk, non_blocking=True)
        self.wv.copy_(layer.wv, non_blocking=True)
        self.wo.copy_(layer.wo, non_blocking=True)
        self.gate_up_proj.copy_(layer.gate_up_proj, non_blocking=True)
        self.down_proj.copy_(layer.down_proj, non_blocking=True)

        self.input_layernorm_weight.copy_(layer.input_layernorm_weight, non_blocking=True)
        self.post_attention_layernorm_weight.copy_(layer.post_attention_layernorm_weight, non_blocking=True)

        self.input_layernorm_variance_epsilon = layer.input_layernorm_variance_epsilon
        self.post_attention_layernorm_variance_epsilon = layer.post_attention_layernorm_variance_epsilon

        self.layer_idx = layer.layer_idx

    def alloc_space(self, layer: GLM4Layer, device):
        self.device = device
        self.wq = torch.zeros_like(layer.wq).to(device)
        self.wk = torch.zeros_like(layer.wk).to(device)
        self.wv = torch.zeros_like(layer.wv).to(device)
        self.wo = torch.zeros_like(layer.wo).to(device)

        self.gate_up_proj = torch.zeros_like(layer.gate_up_proj).to(device)
        self.down_proj = torch.zeros_like(layer.down_proj).to(device)
        self.input_layernorm_weight = torch.zeros_like(layer.input_layernorm_weight).to(device)
        self.post_attention_layernorm_weight = torch.zeros_like(layer.post_attention_layernorm_weight).to(device)
