from __future__ import annotations

from typing import Generic, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    def state_dict(
        self, 
        *, 
        prefix: str = "", 
        result: _STATE_DICT | None = None
    ) -> _STATE_DICT:
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix    : str  = "",
        _internal : bool = False,
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape, (
                    f"shape mismatch for {_concat_prefix(prefix, name)}: "
                    f"{param.shape} vs {item.shape}"
                )
                # Cast to the parameter's dtype so mixed-dtype checkpoints (e.g. fp32
                # GatedDeltaNet A_log / norm.weight) load without an exact-dtype match.
                setattr(self, name, item.to(param.dtype))
            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


T = TypeVar("T", bound=BaseOP)
class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: list[T]):
        self.op_list = ops

    def state_dict(
        self, 
        *, 
        prefix: str = "", 
        result: _STATE_DICT | None = None
    ) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix   : str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
