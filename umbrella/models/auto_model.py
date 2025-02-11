from .llama import Llama, LlamaAwq, LlamaOffload, LlamaAwqOffload, LlamaCudagraph
from .qwen import Qwen, QwenOffload, QwenAwq, QwenAwqOffload, QwenCudagraph
from .mistral import Mistral, MistralAwq, MistralOffload, MistralAwqOffload, MistralCudagraph
class AutoModelLM:
    """
    自动模型加载器，根据模型类型动态加载对应的类。
    """
    _OFFLOAD_MODEL_MAPPING = {
        "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4": LlamaAwqOffload,
        "lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit": LlamaAwqOffload,
        "casperhansen/llama-3.3-70b-instruct-awq": LlamaAwqOffload,
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4": LlamaAwqOffload,
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": LlamaAwqOffload,
        "casperhansen/deepseek-r1-distill-llama-70b-awq": LlamaAwqOffload,
        "meta-llama/Llama-3.3-70B-Instruct": LlamaOffload,
        "meta-llama/Llama-3.1-70B-Instruct": LlamaOffload,
        "meta-llama/Llama-3.1-8B-Instruct": LlamaOffload,
        "meta-llama/Meta-Llama-3-70B-Instruct": LlamaOffload,
        "meta-llama/Meta-Llama-3-8B-Instruct": LlamaOffload,
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":LlamaOffload,
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":LlamaOffload,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":QwenOffload,
        "Qwen/Qwen2.5-Coder-72B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-Coder-32B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-Coder-14B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-Coder-7B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-Coder-3B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-Coder-1.5B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-Coder-0.5B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-0.5B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-1.5B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-3B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-7B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-14B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-32B-Instruct": QwenOffload,
        "Qwen/Qwen2.5-72B-Instruct": QwenOffload,
        "Qwen/QwQ-32B-Preview": QwenOffload,
        "Qwen/Qwen2.5-Coder-72B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-0.5B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-3B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-7B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-14B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-32B-Instruct-AWQ": QwenAwqOffload,
        "Qwen/Qwen2.5-72B-Instruct-AWQ": QwenAwqOffload,
        "KirillR/QwQ-32B-Preview-AWQ": QwenAwqOffload,
        "casperhansen/deepseek-r1-distill-qwen-32b-awq":QwenAwqOffload,
        "mistralai/Mistral-7B-v0.3": MistralOffload,   # Mistral 7B added by EJ
    }
    
    _MODEL_MAPPING = {
        "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4": LlamaAwq,
        "lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit": LlamaAwq,
        "casperhansen/llama-3.3-70b-instruct-awq": LlamaAwq,
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4": LlamaAwq,
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": LlamaAwq,
         "casperhansen/deepseek-r1-distill-llama-70b-awq": LlamaAwq,
        "meta-llama/Llama-3.3-70B-Instruct": Llama,
        "meta-llama/Llama-3.1-70B-Instruct": Llama,
        "meta-llama/Llama-3.1-8B-Instruct": Llama,
        "meta-llama/Meta-Llama-3-70B-Instruct": Llama,
        "meta-llama/Meta-Llama-3-8B-Instruct": Llama,
        "meta-llama/Llama-3.2-1B-Instruct": Llama,
        "meta-llama/Llama-3.2-3B-Instruct": Llama,
        "Felladrin/Llama-68M-Chat-v1": Llama,
        "facebook/layerskip-llama3.2-1B": Llama,
        "Zhuominc/Llama-3-330M": Llama,
        "Zhuominc/Coder-670M": Llama,
        "Zhuominc/Coder-400M": Llama,
        "Zhuominc/Coder-400M-IT": Llama,
        "Zhuominc/FastCode-500M": Llama,
        "InfiniAILab/CodeDrafter-500M": Llama,
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":Llama,
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":Llama,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":Qwen,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":Qwen,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":Qwen,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":Qwen,
        "Qwen/Qwen2.5-Coder-72B-Instruct": Qwen,
        "Qwen/Qwen2.5-Coder-32B-Instruct": Qwen,
        "Qwen/Qwen2.5-Coder-14B-Instruct": Qwen,
        "Qwen/Qwen2.5-Coder-7B-Instruct": Qwen,
        "Qwen/Qwen2.5-Coder-3B-Instruct": Qwen,
        "Qwen/Qwen2.5-Coder-1.5B-Instruct": Qwen,
        "Qwen/Qwen2.5-Coder-0.5B-Instruct": Qwen,
        "Qwen/Qwen2.5-0.5B-Instruct": Qwen,
        "Qwen/Qwen2.5-1.5B-Instruct": Qwen,
        "Qwen/Qwen2.5-3B-Instruct": Qwen,
        "Qwen/Qwen2.5-7B-Instruct": Qwen,
        "Qwen/Qwen2.5-14B-Instruct": Qwen,
        "Qwen/Qwen2.5-32B-Instruct": Qwen,
        "Qwen/Qwen2.5-72B-Instruct": Qwen,
        "Qwen/QwQ-32B-Preview": Qwen,
        "Qwen/Qwen2.5-Coder-72B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-0.5B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-3B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-7B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-14B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-32B-Instruct-AWQ": QwenAwq,
        "Qwen/Qwen2.5-72B-Instruct-AWQ": QwenAwq,
        "KirillR/QwQ-32B-Preview-AWQ": QwenAwq,
        "casperhansen/deepseek-r1-distill-qwen-32b-awq":QwenAwq,
        "mistralai/Mistral-7B-v0.3": Mistral,   # Mistral 7B added by EJ
    }

    _CUDAGRAPH_MODEL_MAPPING = {
        
        "meta-llama/Llama-3.1-8B-Instruct": LlamaCudagraph,
        "meta-llama/Meta-Llama-3-8B-Instruct": LlamaCudagraph,
        "meta-llama/Llama-3.2-1B-Instruct": LlamaCudagraph,
        "meta-llama/Llama-3.2-3B-Instruct": LlamaCudagraph,
        "Felladrin/Llama-68M-Chat-v1": LlamaCudagraph,
        "facebook/layerskip-llama3.2-1B": LlamaCudagraph,
        "Zhuominc/Llama-3-330M": LlamaCudagraph,
        "Zhuominc/Coder-670M": LlamaCudagraph,
        "Zhuominc/Coder-400M": LlamaCudagraph,
        "Zhuominc/Coder-400M-IT": LlamaCudagraph,
        "Zhuominc/FastCode-500M": LlamaCudagraph,
        "InfiniAILab/CodeDrafter-500M": LlamaCudagraph,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":QwenCudagraph,
        "Qwen/Qwen2.5-Coder-72B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-Coder-32B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-Coder-14B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-Coder-7B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-Coder-3B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-Coder-1.5B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-Coder-0.5B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-0.5B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-1.5B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-3B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-7B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-14B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-32B-Instruct": QwenCudagraph,
        "Qwen/Qwen2.5-72B-Instruct": QwenCudagraph,
        "Qwen/QwQ-32B-Preview": QwenCudagraph, 
        "mistralai/Mistral-7B-v0.3": MistralCudagraph,   # Mistral 7B added by EJ
    }
    
    @classmethod
    def from_pretrained(cls, model_name, offload=False, cuda_graph=False, **kwargs):
        """
        根据模型类型加载预训练模型。

        :param model_name: 模型类型，例如 'llama' 或 'gpt'
        :param kwargs: 额外参数
        :return: 对应的模型实例
        """
        if cuda_graph:
            if model_name not in cls._CUDAGRAPH_MODEL_MAPPING:
                raise ValueError(f"Model type '{model_name}' is not supported. "
                                f"Supported types: {list(cls._CUDAGRAPH_MODEL_MAPPING.keys())}")
            model_class = cls._CUDAGRAPH_MODEL_MAPPING[model_name]
            return model_class(model_name = model_name, **kwargs)
        if not offload:
            if model_name not in cls._MODEL_MAPPING:
                raise ValueError(f"Model type '{model_name}' is not supported. "
                                f"Supported types: {list(cls._MODEL_MAPPING.keys())}")
            model_class = cls._MODEL_MAPPING[model_name]
            return model_class(model_name = model_name, **kwargs)
        else:
            if model_name not in cls._OFFLOAD_MODEL_MAPPING:
                raise ValueError(f"Model type '{model_name}' is not supported (offload). "
                                f"Supported (offload) types: {list(cls._OFFLOAD_MODEL_MAPPING.keys())}")
            model_class = cls._OFFLOAD_MODEL_MAPPING[model_name]
            return model_class(model_name = model_name, **kwargs)