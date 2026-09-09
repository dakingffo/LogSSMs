from __future__ import annotations

import glob
import json
import os
import re
from collections.abc import Collection
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from picosgl.distributed import get_tp_info
from picosgl.utils import div_ceil, load_model_config, resolve_model_path
from tqdm import tqdm

_SPLIT_DIM_0 = [
    ".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj", ".gate_up_proj",
    # Hybrid gated-delta attention / MTP
    ".in_proj_qkv", ".in_proj_z", ".in_proj_b", ".in_proj_a",
    ".conv1d.weight",
    ".A_log", ".dt_bias",  # per-value-head vectors, column-parallel like in_proj_b/a
    ".fc",
]
_SPLIT_DIM_1 = [".o_proj", ".down_proj", ".out_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _checkpoint_files(model_folder: str) -> tuple[str, list[str]]:
    """Return the preferred checkpoint format and its ordered shard paths."""
    candidates = (
        ("safetensors", "model.safetensors.index.json", "model.safetensors"),
        ("pytorch", "pytorch_model.bin.index.json", "pytorch_model.bin"),
    )
    for checkpoint_format, index_name, weight_name in candidates:
        index_path = os.path.join(model_folder, index_name)
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as file:
                index = json.load(file)
            filenames = dict.fromkeys(index["weight_map"].values())
            return checkpoint_format, [
                os.path.join(model_folder, filename) for filename in filenames
            ]

        weight_path = os.path.join(model_folder, weight_name)
        if os.path.isfile(weight_path):
            return checkpoint_format, [weight_path]

        pattern = (
            "model-*.safetensors"
            if checkpoint_format == "safetensors"
            else "pytorch_model-*.bin"
        )
        paths = sorted(glob.glob(os.path.join(model_folder, pattern)))
        if paths:
            return checkpoint_format, paths

    # Some converted checkpoints use non-standard safetensors names. Preserve support
    # for them, but prefer the standard model files above and ignore auxiliary exports.
    paths = sorted(glob.glob(os.path.join(model_folder, "*.safetensors")))
    filtered = [path for path in paths if not path.endswith("consolidated.safetensors")]
    if filtered or paths:
        return "safetensors", filtered or paths

    raise FileNotFoundError(
        f"No safetensors or PyTorch checkpoint found in {model_folder}"
    )


def iter_checkpoint_weights(
    model_path   : str,
    device       : torch.device | str = "cpu",
    *,
    names        : Collection[str] | None = None,
    show_progress: bool = False,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield raw checkpoint tensors without model-specific name or layout changes.

    Safetensors and PyTorch ``pytorch_model*.bin`` checkpoints, including their sharded
    index formats, share this interface. Safetensors is preferred when both are present.
    """
    model_folder = resolve_model_path(model_path)
    checkpoint_format, files = _checkpoint_files(model_folder)
    selected_names = set(names) if names is not None else None

    for path in tqdm(files, desc="Loading weights", disable=not show_progress):
        if checkpoint_format == "safetensors":
            with safetensors.safe_open(path, framework="pt", device=str(device)) as file:
                for name in file.keys():
                    if selected_names is None or name in selected_names:
                        yield name, file.get_tensor(name)
            continue

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        for name, tensor in checkpoint.items():
            if selected_names is not None and name not in selected_names:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Checkpoint entry {name!r} in {path} is not a tensor"
                )
            yield name, tensor.to(device)
        del checkpoint


def _shard_tensor(
    key: str,
    value: torch.Tensor,
    r: int,
    n: int,
    num_kv_heads: int,
    qkv_regions: tuple[int, int, int] | None = None,
):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy.

    ``qkv_regions`` = (q_rows, k_rows, v_rows) for the fused linear-attention
    ``in_proj_qkv`` = [q; k; v] (regions of different sizes). A naive dim-0 chunk puts the
    chunk boundary inside a region and makes the layer's local [q;k;v] split mix regions,
    so each region is chunked by head count independently and the local halves re-concat.
    """
    if qkv_regions is not None and (key.count(".in_proj_qkv") or key.count(".conv1d.weight")):
        q_rows, k_rows, v_rows = qkv_regions
        q, k, v = torch.split(value, [q_rows, k_rows, v_rows], dim=0)
        return torch.cat(
            [q.chunk(n, 0)[r], k.chunk(n, 0)[r], v.chunk(n, 0)[r]], dim=0
        ).clone()
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(
    model_path   : str,
    device       : torch.device,
    *,
    skip_prefixes: tuple[str, ...] = (),
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory is one tensor for safetensors or one PyTorch shard,
    plus the model-specific merge buffers."""
    from .register import make_model_config

    model_folder = resolve_model_path(model_path)
    config = make_model_config(load_model_config(model_folder))
    tp_info = get_tp_info()

    # Fused linear-attention in_proj_qkv = [q; k; v] with different region sizes; naive
    # dim-0 sharding would put the tp chunk boundary inside a region. Compute the region
    # sizes so _shard_tensor can shard each by head count.
    qkv_regions = None
    if config.is_hybrid and config.linear_num_key_heads:
        qk = config.linear_num_key_heads * config.linear_key_head_dim
        v = config.linear_num_value_heads * config.linear_value_head_dim
        qkv_regions = (qk, qk, v)

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    weights = iter_checkpoint_weights(
        model_folder,
        device,
        show_progress=tp_info.is_primary(),
    )
    for checkpoint_name, raw in weights:
        # Skip vision/projector weights from multimodal wrappers.
        if checkpoint_name.startswith(
            ("vision_tower.", "multi_modal_projector.", "model.visual.")
        ):
            continue
        name = checkpoint_name
        name = name.removeprefix("language_model.")
        # Qwen3.5: text weights live under model.language_model.*.
        name = name.replace("model.language_model.", "model.", 1)
        if name.startswith(skip_prefixes):
            continue
        tensor = _shard_tensor(
            name, raw, tp_info.rank, tp_info.size, config.num_kv_heads, qkv_regions
        )
        del raw

        info = _get_merge_info(name)
        if info is None:
            out = (name, tensor)
        else:
            merged_key, slot, all_slots = info
            merge_buf.setdefault(merged_key, {})[slot] = tensor
            if not all(s in merge_buf[merged_key] for s in all_slots):
                continue
            parts = [merge_buf[merged_key][s] for s in all_slots]
            del merge_buf[merged_key]
            out = (merged_key, torch.cat(parts, dim=0))

        if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
            packed_key, expert_idx = expert_info
            slots = expert_buf.setdefault(packed_key, {})
            slots[expert_idx] = out[1]
            if len(slots) != config.num_experts:
                continue
            experts = [slots[idx] for idx in range(config.num_experts)]
            del expert_buf[packed_key]
            yield packed_key, torch.stack(experts, dim=0)
        else:  # Normal dense model
            yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"


def load_target_weight(
    model_path: str,
    device    : torch.device,
) -> Iterator[Tuple[str, torch.Tensor]]:
    yield from load_weight(
        model_path,
        device,
        skip_prefixes=("mtp.", "model.mtp."),
    )
