from __future__ import annotations

from dataclasses import dataclass

import torch


LOG_STATE_GRANULARITY = 16
DEFAULT_LOG_STATE_PAGE_SIZE = 64


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class LogStateSlotLayout:
    page_size        : int
    num_linear_layers: int
    num_key_heads    : int
    num_value_heads  : int
    head_k_dim       : int
    head_v_dim       : int
    conv_dim         : int
    state_len        : int
    dtype            : torch.dtype
    granularity      : int = LOG_STATE_GRANULARITY

    def __post_init__(self) -> None:
        assert self.page_size > 0
        assert self.granularity > 0
        assert self.page_size % self.granularity == 0, (
            f"LogState page size {self.page_size} must be divisible by "
            f"the {self.granularity}-token storage granularity."
        )
        assert self.num_linear_layers > 0
        assert self.num_key_heads > 0
        assert self.num_value_heads > 0
        assert self.num_key_heads == self.num_value_heads, (
            "The in-place LogState layout requires one key head per value head."
        )
        assert self.head_k_dim > 0
        assert self.head_v_dim > 0
        assert self.conv_dim > 0
        assert self.state_len > 0

    @property
    def num_granules(self) -> int:
        return self.page_size // self.granularity

    @property
    def log_key_numel(self) -> int:
        return (
            self.num_linear_layers
            * self.page_size
            * self.num_key_heads
            * self.head_k_dim
        )

    @property
    def log_value_numel(self) -> int:
        return (
            self.num_linear_layers
            * self.page_size
            * self.num_value_heads
            * self.head_v_dim
        )

    @property
    def log_decay_numel(self) -> int:
        return self.num_linear_layers * self.page_size * self.num_value_heads

    @property
    def recurrent_state_numel(self) -> int:
        return (
            self.num_linear_layers
            * self.num_value_heads
            * self.head_k_dim
            * self.head_v_dim
        )

    @property
    def conv_state_numel(self) -> int:
        return (
            self.num_linear_layers
            * self.num_granules
            * self.conv_dim
            * self.state_len
        )

    @property
    def log_key_bytes(self) -> int:
        return self.log_key_numel * self.dtype.itemsize

    @property
    def head_log_key_bytes(self) -> int:
        return self.page_size * self.head_k_dim * self.dtype.itemsize

    @property
    def head_log_value_offset(self) -> int:
        return self.head_log_key_bytes

    @property
    def head_log_value_bytes(self) -> int:
        return self.page_size * self.head_v_dim * self.dtype.itemsize

    @property
    def log_value_bytes(self) -> int:
        return self.log_value_numel * self.dtype.itemsize

    @property
    def head_log_decay_offset(self) -> int:
        return _align_up(
            self.head_log_value_offset + self.head_log_value_bytes,
            torch.float32.itemsize,
        )

    @property
    def head_log_decay_bytes(self) -> int:
        return self.page_size * torch.float32.itemsize

    @property
    def log_decay_bytes(self) -> int:
        return self.log_decay_numel * torch.float32.itemsize

    @property
    def head_log_payload_bytes(self) -> int:
        return self.head_log_decay_offset + self.head_log_decay_bytes

    @property
    def head_recurrent_state_bytes(self) -> int:
        return self.head_k_dim * self.head_v_dim * self.dtype.itemsize

    @property
    def recurrent_state_bytes(self) -> int:
        return self.recurrent_state_numel * self.dtype.itemsize

    @property
    def payload_bytes(self) -> int:
        head_payload_bytes = _align_up(
            max(
                self.head_log_payload_bytes,
                self.head_recurrent_state_bytes,
            ),
            max(self.dtype.itemsize, torch.float32.itemsize),
        )
        return (
            self.num_linear_layers
            * self.num_value_heads
            * head_payload_bytes
        )

    @property
    def head_payload_bytes(self) -> int:
        return self.payload_bytes // (
            self.num_linear_layers * self.num_value_heads
        )

    @property
    def layer_payload_bytes(self) -> int:
        return self.num_value_heads * self.head_payload_bytes

    @property
    def conv_state_offset(self) -> int:
        return self.payload_bytes

    @property
    def conv_state_bytes(self) -> int:
        return self.conv_state_numel * self.dtype.itemsize

    @property
    def slot_bytes(self) -> int:
        return _align_up(
            self.conv_state_offset + self.conv_state_bytes,
            max(self.dtype.itemsize, torch.float32.itemsize),
        )


@dataclass(frozen=True)
class LinearLogView:
    key           : torch.Tensor
    residual_value: torch.Tensor
    decay         : torch.Tensor


@dataclass(frozen=True)
class LinearLogGranule:
    log       : LinearLogView
    conv_state: torch.Tensor


@dataclass(frozen=True)
class LinearCacheSlot:
    log            : LinearLogView
    recurrent_state: torch.Tensor
    conv_state     : torch.Tensor


class LogStatePool:
    """Physical storage for page-sized linear-attention cache slots.

    A slot covers ``page_size`` tokens. Each layer/head owns an independent
    payload whose log representation stores realized low-rank updates and whose
    checkpoint representation stores one recurrent state in the same bytes.
    This head-local overlay lets one CUDA block consume a complete log before it
    overwrites that payload in place. Convolution snapshots are common side data
    and are retained at every ``granularity``-token boundary.

    The allocation and compression unit is one complete slot. Granules only make
    an admitted log page recoverable at finer boundaries; a partially-filled slot
    must never be admitted into the prefix cache.

    Allocation, ownership, representation tags and prefix ranges intentionally do
    not live here. They belong to the cache manager and Log-Radix-Tree.
    """

    def __init__(
        self,
        num_slots        : int,
        page_size        : int,
        num_linear_layers: int,
        num_key_heads    : int,
        num_value_heads  : int,
        head_k_dim       : int,
        head_v_dim       : int,
        conv_dim         : int,
        state_len        : int,
        device           : torch.device,
        dtype            : torch.dtype,
        granularity      : int = LOG_STATE_GRANULARITY,
    ) -> None:
        assert num_slots > 0
        self.layout = LogStateSlotLayout(
            page_size=page_size,
            num_linear_layers=num_linear_layers,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_dim=conv_dim,
            state_len=state_len,
            dtype=dtype,
            granularity=granularity,
        )
        self._storage = torch.empty(
            num_slots, self.layout.slot_bytes,
            dtype=torch.uint8, device=device,
        )
        self._num_slots = num_slots
        self._device = device
        self._dtype = dtype

        L = num_linear_layers
        P = page_size
        H = num_value_heads
        G = self.layout.num_granules
        head_stride = self.layout.head_payload_bytes
        layer_stride = self.layout.layer_payload_bytes
        self.key_log = self._strided_view(
            offset=0,
            shape=(L, P, H, head_k_dim),
            strides=(
                layer_stride,
                head_k_dim * self._dtype.itemsize,
                head_stride,
                self._dtype.itemsize,
            ),
            dtype=dtype,
        )
        self.residual_value_log = self._strided_view(
            offset=self.layout.head_log_value_offset,
            shape=(L, P, H, head_v_dim),
            strides=(
                layer_stride,
                head_v_dim * self._dtype.itemsize,
                head_stride,
                self._dtype.itemsize,
            ),
            dtype=dtype,
        )
        self.decay_log = self._strided_view(
            offset=self.layout.head_log_decay_offset,
            shape=(L, P, H),
            strides=(layer_stride, torch.float32.itemsize, head_stride),
            dtype=torch.float32,
        )
        self.recurrent_state = self._strided_view(
            offset=0,
            shape=(L, H, head_k_dim, head_v_dim),
            strides=(
                layer_stride,
                head_stride,
                head_v_dim * self._dtype.itemsize,
                self._dtype.itemsize,
            ),
            dtype=dtype,
        )
        self.conv_state = self._view(
            offset=self.layout.conv_state_offset,
            shape=(L, G, conv_dim, state_len),
            dtype=dtype,
        )

    def _view(
        self,
        offset: int,
        shape : tuple[int, ...],
        dtype : torch.dtype,
    ) -> torch.Tensor:
        numel = 1
        for dim in shape:
            numel *= dim
        num_bytes = numel * dtype.itemsize
        assert offset % dtype.itemsize == 0
        assert offset + num_bytes <= self.layout.slot_bytes
        return self._storage[:, offset : offset + num_bytes].view(dtype).view(
            self.num_slots, *shape
        )

    def _strided_view(
        self,
        offset : int,
        shape  : tuple[int, ...],
        strides: tuple[int, ...],
        dtype  : torch.dtype,
    ) -> torch.Tensor:
        itemsize = dtype.itemsize
        assert offset % itemsize == 0
        assert self.layout.slot_bytes % itemsize == 0
        assert all(stride % itemsize == 0 for stride in strides)
        storage = self._storage.view(dtype)
        return torch.as_strided(
            storage,
            size=(self.num_slots, *shape),
            stride=(
                self.layout.slot_bytes // itemsize,
                *(stride // itemsize for stride in strides),
            ),
            storage_offset=offset // itemsize,
        )

    def slot(self, index: int) -> LinearCacheSlot:
        self._check_slot(index)
        return LinearCacheSlot(
            log=LinearLogView(
                key=self.key_log[index],
                residual_value=self.residual_value_log[index],
                decay=self.decay_log[index],
            ),
            recurrent_state=self.recurrent_state[index],
            conv_state=self.conv_state[index],
        )

    def granule(self, slot: int, index: int) -> LinearLogGranule:
        self._check_slot(slot)
        if not 0 <= index < self.num_granules:
            raise IndexError(
                f"LogState granule {index} is outside [0, {self.num_granules})."
            )
        begin = index * self.layout.granularity
        end = begin + self.layout.granularity
        return LinearLogGranule(
            log=LinearLogView(
                key=self.key_log[slot, :, begin:end],
                residual_value=self.residual_value_log[slot, :, begin:end],
                decay=self.decay_log[slot, :, begin:end],
            ),
            conv_state=self.conv_state[slot, :, index],
        )

    def token_granule_index(self, position: int) -> int:
        if not 0 <= position < self.page_size:
            raise ValueError(
                f"Token position {position} is outside [0, {self.page_size})."
            )
        return position // self.layout.granularity

    def recovery_granule_index(self, position: int) -> int:
        if not 0 < position <= self.page_size:
            raise ValueError(
                f"Recovery position {position} is outside "
                f"the slot interval (0, {self.page_size}]."
            )
        if position % self.layout.granularity != 0:
            raise ValueError(
                f"Recovery position {position} is not aligned to the "
                f"{self.layout.granularity}-token recovery granularity."
            )
        return position // self.layout.granularity - 1

    def _check_slot(self, index: int) -> None:
        if not 0 <= index < self.num_slots:
            raise IndexError(
                f"Linear cache slot {index} is outside [0, {self.num_slots})."
            )

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def page_size(self) -> int:
        return self.layout.page_size

    @property
    def num_granules(self) -> int:
        return self.layout.num_granules

    @property
    def slot_bytes(self) -> int:
        return self.layout.slot_bytes

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
