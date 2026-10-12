# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import dataclasses
import enum
import json
import logging
import os
import re
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import torch
import triton
from packaging import version
from triton.runtime.autotuner import Autotuner

TRITON_ABOVE_3_5_1 = version.parse(triton.__version__) >= version.parse("3.5.1")
TRITON_ABOVE_3_4_0 = version.parse(triton.__version__) >= version.parse("3.4.0")


class FlaCacheMode(enum.Enum):
    DISABLED = "disabled"
    STRICT = "strict"
    FUZZY = "fuzzy"
    FULL = "full"
    DEFAULT = "default"
    ALWAYS = "always"


    @classmethod
    def from_env(cls) -> "FlaCacheMode":
        mode_str = os.environ.get("FLA_CACHE_MODE", cls.DISABLED.value)
        try:
            return cls(mode_str)
        except ValueError:
            valid = [m.value for m in cls]
            raise ValueError(
                f"Invalid FLA_CACHE_MODE={mode_str!r}. Valid values: {valid}"
            ) from None


FLA_CACHE_MODE: FlaCacheMode = FlaCacheMode.from_env()
logger = logging.getLogger(__name__)


def sanitize_gpu_name(gpu_name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", gpu_name)
    sanitized = sanitized.strip("_")
    return sanitized or "unknown_gpu"


@lru_cache(maxsize=1)
def get_gpu_info():

    gpu_name = None

    if "FLA_GPU_NAME" in os.environ:
        gpu_name = os.environ["FLA_GPU_NAME"]

    elif torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        gpu_name = torch.xpu.get_device_name(0)

    if gpu_name:
        return sanitize_gpu_name(gpu_name)


    return "cpu"


def get_fla_config_dir() -> Path:

    if "FLA_CONFIG_DIR" in os.environ:
        return Path(os.environ["FLA_CONFIG_DIR"])


    project_dir = Path(__file__).parent.parent.parent
    return project_dir / "configs" / get_gpu_info()


@dataclasses.dataclass(frozen=True)
class AutotuneKey:
    autotune_key: tuple[Any, ...]

    @staticmethod
    def normalize_autotune_key(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [AutotuneKey.normalize_autotune_key(v) for v in value]
        if isinstance(value, dict):
            return {k: AutotuneKey.normalize_autotune_key(v) for k, v in value.items()}
        return value

    @staticmethod
    def serialize(key: Any) -> str:
        return json.dumps(AutotuneKey.normalize_autotune_key(key), separators=(",", ":"), sort_keys=True)

    @staticmethod
    def key_hash(key: Any) -> str:
        import hashlib
        return hashlib.md5(AutotuneKey.serialize(key).encode()).hexdigest()

    @staticmethod
    def is_numeric(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def keys_fuzzy_match(cached_key: Any, requested_key: Any) -> bool:


        if AutotuneKey.is_numeric(cached_key) and AutotuneKey.is_numeric(requested_key):
            return True
        if isinstance(cached_key, (list, tuple)) and isinstance(requested_key, (list, tuple)):
            return len(cached_key) == len(requested_key) and all(
                AutotuneKey.keys_fuzzy_match(c, r) for c, r in zip(cached_key, requested_key)
            )
        if isinstance(cached_key, dict) and isinstance(requested_key, dict):
            return cached_key.keys() == requested_key.keys() and all(
                AutotuneKey.keys_fuzzy_match(cached_key[k], requested_key[k]) for k in cached_key
            )
        return cached_key == requested_key

    @classmethod
    def build(
        cls,
        arg_names: list[str],
        key_names: list[str],
        positional_args: tuple[Any, ...],
        runtime_kwargs: dict[str, Any],
    ) -> "AutotuneKey":
        named_args = dict(zip(arg_names, positional_args))
        all_args = {**named_args, **runtime_kwargs}
        tracked_args = {k: v for (k, v) in all_args.items() if k in arg_names}
        tuning_key = [tracked_args[name] for name in key_names if name in tracked_args]
        for arg in tracked_args.values():
            if hasattr(arg, "dtype"):
                tuning_key.append(str(arg.dtype))
        return cls(autotune_key=tuple(tuning_key))


    def fuzzy_matches(self, entry_key: Any) -> bool:
        self_normalized = self.normalize_autotune_key(self.autotune_key)
        entry_normalized = self.normalize_autotune_key(entry_key)
        return (
            isinstance(self_normalized, list)
            and isinstance(entry_normalized, list)
            and len(self_normalized) == len(entry_normalized)
            and AutotuneKey.keys_fuzzy_match(self_normalized, entry_normalized)
        )


@dataclasses.dataclass(frozen=True)
class KernelConfigFile:
    kernel_name: str | None
    triton_version: str | None
    autotune_entries: dict[str, dict[str, Any]] | None
    default_config: dict[str, Any] | None

    @classmethod
    def from_dict(cls, config_file: Path, data: Any) -> "KernelConfigFile | None":
        def fail(msg, *args):
            logger.warning(msg, *args)
            raise ValueError

        try:
            if not isinstance(data, dict):
                fail("Malformed config %s: root is %s, expected dict", config_file, type(data).__name__)
            raw_entries = data.get("autotune_entries")
            entries: dict[str, dict[str, Any]] | None = None
            if raw_entries is not None:
                if not isinstance(raw_entries, dict):
                    fail("Malformed config %s: 'autotune_entries' is %s, expected dict",
                         config_file, type(raw_entries).__name__)
                for h, entry in raw_entries.items():
                    if not isinstance(entry, dict):
                        fail("Malformed config %s: autotune_entries[%r] is %s, expected dict",
                             config_file, h, type(entry).__name__)
                    if not isinstance(entry.get("config"), dict):
                        fail("Malformed config %s: autotune_entries[%r] missing valid 'config' field", config_file, h)
                entries = raw_entries
            default_config = data.get("default_config")
            if default_config is not None and not isinstance(default_config, dict):
                fail("Malformed config %s: 'default_config' is %s, expected dict", config_file, type(default_config).__name__)
            return cls(
                kernel_name=data.get("kernel_name"),
                triton_version=data.get("triton_version"),
                autotune_entries=entries,
                default_config=default_config,
            )
        except ValueError:
            return None


    def lookup_exact(self, key: AutotuneKey) -> dict[str, Any] | None:
        if self.autotune_entries is None:
            return None
        return self.autotune_entries.get(AutotuneKey.key_hash(key.autotune_key))

    def lookup_fuzzy(self, key: AutotuneKey) -> dict[str, Any] | None:
        if self.autotune_entries is None:
            return None
        for entry in self.autotune_entries.values():
            if key.fuzzy_matches(entry.get("autotune_key")):
                return entry
        return None


@cache
def load_config_file(config_file: Path) -> dict[str, Any] | None:
    try:
        with open(config_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Error reading config file %s: %s", config_file, e)
        return None


def read_config_file(config_file: Path) -> dict[str, Any] | None:
    if FLA_CACHE_MODE is FlaCacheMode.ALWAYS:
        return load_config_file.__wrapped__(config_file)
    return load_config_file(config_file)


def load_cached_config(kernel_name: str, autotune_key: AutotuneKey | None = None) -> dict[str, Any] | None:
    if FLA_CACHE_MODE is FlaCacheMode.DISABLED:
        return None

    config_dir = get_fla_config_dir()
    config_file = config_dir / f"{kernel_name}.json"

    if not config_file.exists():
        return None

    config_data = read_config_file(config_file)
    if config_data is None:
        return None
    config = KernelConfigFile.from_dict(config_file, config_data)
    if config is None:
        return None

    if FLA_CACHE_MODE is FlaCacheMode.DEFAULT or FLA_CACHE_MODE is FlaCacheMode.ALWAYS:
        return config.default_config


    if FLA_CACHE_MODE is FlaCacheMode.STRICT:
        if autotune_key is not None:
            entry = config.lookup_exact(autotune_key)
            if entry is not None:
                return entry["config"]
        return None


    if autotune_key is not None:
        entry = config.lookup_exact(autotune_key) or config.lookup_fuzzy(autotune_key)
        if entry is not None:
            return entry["config"]

    if FLA_CACHE_MODE is FlaCacheMode.FUZZY:
        return None


    if config.default_config is not None:
        return config.default_config
    if config.autotune_entries is not None:
        return None
    return config_data


class CachedAutotuner(Autotuner):

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, **kwargs):
        super().__init__(fn, arg_names, configs, key, reset_to_zero, restore_value, **kwargs)
        self.kernel_name = fn.fn.__name__ if hasattr(fn, 'fn') else fn.__name__


        if not self.user_defined_pre_hook and (self.reset_to_zero or self.restore_value):
            def _pre_hook(kw, reset_only=False):
                for n in self.reset_to_zero:
                    if kw[n] is not None:
                        kw[n].zero_()
                if not reset_only:
                    self.restore_copies = {n: kw[n].clone() for n in self.restore_value if kw[n] is not None}
            self.pre_hook = _pre_hook
        if not self.user_defined_post_hook and self.restore_value:
            def _post_hook(kw, exception):
                for n, copy in self.restore_copies.items():
                    kw[n].copy_(copy)
                self.restore_copies = {}
            self.post_hook = _post_hook

    def should_check_fla_cache(self, key: AutotuneKey) -> bool:
        if FLA_CACHE_MODE is FlaCacheMode.DISABLED:
            return False
        if FLA_CACHE_MODE is FlaCacheMode.ALWAYS:
            return True
        return key.autotune_key not in self.cache

    def run(self, *args, **kwargs):
        key = AutotuneKey.build(self.arg_names, self.keys, args, kwargs)
        if self.should_check_fla_cache(key):
            self.maybe_load_cached_config(key)
        return super().run(*args, **kwargs)

    def maybe_load_cached_config(self, key: AutotuneKey):
        best_config = load_cached_config(self.kernel_name, key)

        if best_config is not None:
            kw = best_config["kwargs"]
            num_warps = best_config["num_warps"]
            num_stages = best_config["num_stages"]

            extra = {
                "num_ctas": best_config["num_ctas"],
                "maxnreg": best_config.get("maxnreg"),
                "pre_hook": None,
                "ir_override": best_config.get("ir_override"),
            } if TRITON_ABOVE_3_5_1 else {}
            cfg = triton.Config(kw, num_warps=num_warps, num_stages=num_stages, **extra)

            self.cache[key.autotune_key] = cfg
        else:
            logger.debug(
                "No cached config found for kernel %s and key %s; falling back to Triton autotune",
                self.kernel_name,
                list(key.autotune_key),
            )


def fla_cache_autotune(configs, key=None, prune_configs_by=None, reset_to_zero=None, restore_value=None,
                       pre_hook=None, post_hook=None, warmup=None, rep=None, use_cuda_graph=False,
                       do_bench=None, cache_results=False):

    if key is None:
        key = []

    def decorator(fn):
        kwargs = {}
        if TRITON_ABOVE_3_4_0:
            kwargs = {"cache_results": cache_results}

        return CachedAutotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value,
                               pre_hook=pre_hook, post_hook=post_hook,
                               prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                               use_cuda_graph=use_cuda_graph, do_bench=do_bench,
                               **kwargs,
                               )

    return decorator
