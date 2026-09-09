from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class, make_model_config
from .weight import iter_checkpoint_weights, load_target_weight, load_weight


def make_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)


__all__ = [
    "ModelConfig",
    "RotaryConfig",
    "iter_checkpoint_weights",
    "load_target_weight",
    "load_weight",
    "make_model",
    "make_model_config",
]
