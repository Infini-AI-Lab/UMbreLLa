from .dynamic_speculation_engine import DynamicSpeculationEngine
from .static_speculation_engine import StaticSpeculationEngine

class AutoEngine:
    _ENGINE_MAPPING = {
        'static': StaticSpeculationEngine,
        'dynamic': DynamicSpeculationEngine  
    }
    
    
    @classmethod
    def from_config(cls, device: str, **kwargs):
        engine_name = kwargs.pop("engine", 'dynamic')
        if engine_name not in cls._ENGINE_MAPPING:
                raise ValueError(f"Engine type '{engine_name}' is not supported. "
                                f"Supported types: {list(cls._ENGINE_MAPPING.keys())}")
        engine_class = cls._ENGINE_MAPPING[engine_name]
        draft_model_name = kwargs.pop("draft_model", None)
        target_model_name = kwargs.pop("model", None)
        assert draft_model_name is not None
        assert target_model_name is not None
        return engine_class(draft_model_name=draft_model_name, target_model_name=target_model_name,device=device, **kwargs)