from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from logssms.distributed import DistributedInfo
from logssms.scheduler import SchedulerConfig
from logssms.utils import init_logger
from logssms.utils.args import parse_bool


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host  : str  = "127.0.0.1"
    server_port  : int  = 1919
    num_tokenizer: int  = 1
    silent_output: bool = False

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: list[str]) -> ServerArgs:
    from logssms.layers.attention_backend import validate_attn_backend
    from logssms.cache import (
        SUPPORTED_CACHE_ARGUMENT_PARSERS,
        make_cache_argument_parser,
    )
    from logssms.layers.moe_backend import SUPPORTED_MOE_BACKENDS
    from logssms.speculator.drafters import (
        SUPPORTED_SPECULATOR_ARGUMENT_PARSERS,
        make_speculator_argument_parser,
    )

    selection_parser = argparse.ArgumentParser(add_help=False)
    selection_parser.add_argument(
        "--speculative-algorithm",
        choices=SUPPORTED_SPECULATOR_ARGUMENT_PARSERS.supported_names(),
        default=None,
    )
    selection_parser.add_argument(
        "--cache-type",
        choices=SUPPORTED_CACHE_ARGUMENT_PARSERS.supported_names(),
        default=ServerArgs.cache_type,
    )
    known_args, _ = selection_parser.parse_known_args(args)
    speculator_parser = (
        make_speculator_argument_parser(known_args.speculative_algorithm)
        if known_args.speculative_algorithm is not None else None
    )
    cache_parser = make_cache_argument_parser(known_args.cache_type)
    parents = [cache_parser.parser]
    if speculator_parser is not None:
        parents.append(speculator_parser.parser)
    parser = argparse.ArgumentParser(
        description="logssms Server Arguments",
        parents=parents,
    )

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="A local model directory or a Hugging Face/ModelScope model ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help=(
            "The fraction of initially free GPU memory available to model weights "
            "and runtime caches."
        ),
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help=(
            "Maximum request bucket for decode and mixed CUDA graphs. Also enables "
            "piecewise prefill graphs. 0 disables all graphs; None selects sizes "
            "based on GPU memory."
        ),
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch.",
    )

    parser.add_argument(
        "--max-prefill-tokens",
        type=int,
        dest="max_prefill_tokens",
        default=ServerArgs.max_prefill_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--scheduling-policy",
        choices=["prefill-first", "mixed"],
        default=ServerArgs.scheduling_policy,
        help="Prefill priority or prefill requests mixed with decoding requests.",
    )
    parser.add_argument(
        "--mixed-prefill-tokens",
        type=int,
        default=ServerArgs.mixed_prefill_tokens,
        help="Total prefill token budget per mixed batch: positive multiple of LogState granularity "
             "(64 for standard caches), <= max-prefill-tokens.",
    )

    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        dest="max_decode_tokens",
        default=None,
        help=(
            "Max total tokens (positions) in a single decode/verify batch. "
            "Default: max_running_req for autoregressive decoding, "
            "max_running_req // 2 for speculative decoding."
        ),
    )

    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_ARGUMENT_PARSERS.supported_names(),
        help="The cache management strategy.",
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--speculative-algorithm",
        type=str,
        dest="speculative_algorithm",
        default=None,
        choices=SUPPORTED_SPECULATOR_ARGUMENT_PARSERS.supported_names(),
        help="The speculative decoding algorithm.",
    )

    parser.add_argument(
        "--speculative-draft-model-path",
        type=str,
        dest="speculative_draft_model_path",
        default=ServerArgs.speculative_draft_model_path,
        help="Path of the drafter weights.",
    )

    parser.add_argument(
        "--dt-separation",
        type=parse_bool,
        default=ServerArgs.dt_separation,
        metavar="{true,false}",
        dest="dt_separation",
        help="Run the drafter on a separate device (requires tp_size + 1 GPUs).",
    )


    kwargs = parser.parse_args(args).__dict__.copy()

    kwargs["cache_config"] = cache_parser.make_config(kwargs)
    alignment = kwargs["cache_config"].prefill_alignment
    mixed_tokens = kwargs["mixed_prefill_tokens"]
    if mixed_tokens <= 0 or mixed_tokens % alignment:
        parser.error(f"--mixed-prefill-tokens must be a positive multiple of {alignment}")
    if kwargs["scheduling_policy"] == "mixed" and mixed_tokens > kwargs["max_prefill_tokens"]:
        parser.error("--mixed-prefill-tokens must be <= --max-prefill-tokens for mixed scheduling")
    if kwargs["scheduling_policy"] == "mixed" and kwargs["speculative_algorithm"] is not None:
        parser.error("mixed scheduling does not support speculative decoding")

    assert kwargs["num_tokenizer"] >= 1, "--num-tokenizer must be >= 1"

    if speculator_parser is not None:
        kwargs["speculator_config"] = speculator_parser.make_config(kwargs)


    if kwargs["max_decode_tokens"] is None:
        kwargs["max_decode_tokens"] = (
            kwargs["max_running_req"] // 2
            if kwargs["speculative_algorithm"] is not None else kwargs["max_running_req"]
        )

    from logssms.utils import resolve_model_path

    model_source = kwargs.pop("model_source")
    unresolved_model_path = kwargs["model_path"]
    kwargs["model_path"] = resolve_model_path(
        unresolved_model_path,
        model_source,
        download_weights=not kwargs["use_dummy_weight"],
    )
    if draft_path := kwargs["speculative_draft_model_path"]:
        kwargs["speculative_draft_model_path"] = (
            kwargs["model_path"]
            if draft_path == unresolved_model_path else
            resolve_model_path(
                draft_path,
                model_source,
                download_weights=not kwargs["use_dummy_weight"],
            )
        )

    speculator_config = kwargs.get("speculator_config")
    if (
        speculator_config is not None
        and speculator_config.requires_model_resolution
    ):
        from logssms.models import make_model_config
        from logssms.utils import load_model_config

        model_config = make_model_config(load_model_config(kwargs["model_path"]))
        kwargs["speculator_config"] = speculator_config.resolve(model_config)

    if (dtype_str := kwargs["dtype"]) == "auto":
        from logssms.utils import load_model_config


        pretrained_config = load_model_config(kwargs["model_path"])
        dtype_str = (
            getattr(pretrained_config, "dtype", None)
            or getattr(pretrained_config, "torch_dtype", None)
            or "bfloat16"
        )

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result
