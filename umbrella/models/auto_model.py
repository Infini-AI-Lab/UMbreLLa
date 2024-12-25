from .llama import Llama, LlamaAwq, LlamaOffload, LlamaAwqOffload

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
        "meta-llama/Llama-3.3-70B-Instruct": LlamaOffload,
        "meta-llama/Llama-3.1-70B-Instruct": LlamaOffload,
        "meta-llama/Llama-3.1-8B-Instruct": LlamaOffload,
        "meta-llama/Meta-Llama-3-70B-Instruct": LlamaOffload,
        "meta-llama/Meta-Llama-3-8B-Instruct": LlamaOffload,
    }
    
    _MODEL_MAPPING = {
        "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4": LlamaAwq,
        "lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit": LlamaAwq,
        "casperhansen/llama-3.3-70b-instruct-awq": LlamaAwq,
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4": LlamaAwq,
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": LlamaAwq,
        "meta-llama/Llama-3.3-70B-Instruct": Llama,
        "meta-llama/Llama-3.1-70B-Instruct": Llama,
        "meta-llama/Llama-3.1-8B-Instruct": Llama,
        "meta-llama/Meta-Llama-3-70B-Instruct": Llama,
        "meta-llama/Meta-Llama-3-8B-Instruct": Llama,
        "meta-llama/Llama-3.2-1B-Instruct": Llama,
        "meta-llama/Llama-3.2-3B-Instruct": Llama,
    }

    @classmethod
    def from_pretrained(cls, model_name, offload=False, **kwargs):
        """
        根据模型类型加载预训练模型。

        :param model_name: 模型类型，例如 'llama' 或 'gpt'
        :param kwargs: 额外参数
        :return: 对应的模型实例
        """
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