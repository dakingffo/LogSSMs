import functools
import json
import os
from typing import Any, Literal

from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

ModelSource = Literal["huggingface", "modelscope"]


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(
    model_path: str,
    source    : ModelSource = "huggingface",
) -> PreTrainedTokenizerBase:
    model_path = resolve_model_path(model_path, source, download_weights=False)
    return AutoTokenizer.from_pretrained(model_path)


@functools.cache
def _load_transformers_config(model_path: str) -> Any:
    return AutoConfig.from_pretrained(model_path)


def _load_config_json(model_path: str) -> dict[str, Any]:
    path = os.path.join(model_path, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No config.json in {model_path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _make_config_namespace(config: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    dict_keys = ("rope_parameters", "rope_scaling")
    return SimpleNamespace(
        **{
            key: (
                _make_config_namespace(value)
                if isinstance(value, dict) and key not in dict_keys else
                value
            )
            for key, value in config.items()
        }
    )


def load_model_config(
    model_path: str,
    source    : ModelSource = "huggingface",
) -> PretrainedConfig:
    model_path = resolve_model_path(model_path, source, download_weights=False)
    raw_config = _load_config_json(model_path)
    model_type = raw_config.get("model_type")
    if model_type is not None and model_type not in CONFIG_MAPPING:
        return _make_config_namespace(raw_config)

    try:
        config = _load_transformers_config(model_path)
        return type(config)(**config.to_dict())
    except Exception:
        # Fallback for architectures not yet registered in this transformers version
        # (e.g. Qwen3Next). Builds a lightweight attribute object mirroring the config;
        # picosgl only reads attributes off it, so this is sufficient.
        logger = __import__("logging").getLogger("picosgl.platform")
        logger.warning("AutoConfig failed for %s, falling back to raw config.json", model_path)
        return _make_config_namespace(raw_config)


def resolve_model_path(
    model_path      : str,
    source          : ModelSource = "huggingface",
    *,
    download_weights: bool = True,
) -> str:
    if source not in ("huggingface", "modelscope"):
        raise ValueError(f"Unknown model source: {source!r}")

    model_path = os.path.expanduser(model_path)
    if os.path.isdir(model_path):
        return model_path

    unsupported_weight_patterns = ["*.pt", "*.ckpt", "*.gguf"]
    ignore_patterns = unsupported_weight_patterns + (
        [] if download_weights else ["*.safetensors", "*.bin"]
    )

    try:
        if source == "huggingface":
            from huggingface_hub import snapshot_download

            return snapshot_download(
                model_path,
                ignore_patterns=ignore_patterns,
                tqdm_class=DisabledTqdm,
            )
        else:
            from modelscope import snapshot_download

            return snapshot_download(model_path, ignore_patterns=ignore_patterns)
    except Exception as e:
        raise ValueError(
            f"Model path {model_path!r} is neither a local directory nor a valid "
            f"{source} model ID: {e}"
        )
