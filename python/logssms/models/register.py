import importlib

from transformers import PretrainedConfig

from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM"                 : (".llama", "LlamaConfig", "LlamaForCausalLM"),
    "Qwen2ForCausalLM"                 : (".qwen2", "Qwen2Config", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM"                 : (".qwen3", "Qwen3Config", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM"              : (".qwen3", "Qwen3MoeConfig", "Qwen3ForCausalLM"),
    "Qwen3_5ForConditionalGeneration"  : (".qwen3_next", "Qwen3_5Config", "Qwen3NextForCausalLM"),
    "Qwen3_5ForCausalLM"               : (".qwen3_next", "Qwen3_5Config", "Qwen3NextForCausalLM"),
    "Qwen3_5MoeForConditionalGeneration": (".qwen3_next", "Qwen3_5MoeConfig", "Qwen3NextForCausalLM"),
    "Qwen3_5MoeForCausalLM"            : (".qwen3_next", "Qwen3_5MoeConfig", "Qwen3NextForCausalLM"),
    "Qwen3NextForCausalLM"             : (".qwen3_next", "Qwen3NextConfig", "Qwen3NextForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, _, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


def make_model_config(config: PretrainedConfig) -> ModelConfig:
    top_architectures = getattr(config, "architectures", None)
    text = getattr(config, "text_config", None) or config
    architectures = top_architectures or getattr(text, "architectures", None)
    if not architectures:
        raise ValueError("Model config does not define an architecture")

    architecture = architectures[0]
    if architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {architecture} not supported")
    module_path, config_name, _ = _MODEL_REGISTRY[architecture]
    module = importlib.import_module(module_path, package=__package__)
    config_cls = getattr(module, config_name)
    return config_cls.from_pretrained(config)


__all__ = ["get_model_class", "make_model_config"]
