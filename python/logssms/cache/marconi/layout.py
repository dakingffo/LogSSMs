from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class HybridBlockLayout:
    page_size: int
    layers_per_group: int
    state_groups: int
    layer_elements: int
    kv_heads: int
    head_dim: int
    state_shape: tuple[int, int, int]
    conv_shape: tuple[int, int]

    @classmethod
    def make(cls, attention_layers, linear_spec, kv_heads, head_dim, page_size=0):
        if attention_layers < 1:
            raise ValueError("Marconi requires at least one full-attention layer")
        state_elements = prod(linear_spec.state_shape) + prod(linear_spec.conv_shape)
        token_elements = 2 * kv_heads * head_dim
        minimum = ((state_elements + token_elements * 64 - 1) // (token_elements * 64)) * 64
        if page_size and (page_size < minimum or page_size % 64):
            raise ValueError(f"Marconi page size must be a multiple of 64 and at least {minimum}")
        page_size = page_size or minimum
        return cls(page_size, attention_layers,
                   (linear_spec.num_layers + attention_layers - 1) // attention_layers,
                   page_size * token_elements, kv_heads, head_dim,
                   linear_spec.state_shape, linear_spec.conv_shape)

    def block_bytes(self, itemsize: int) -> int:
        return self.layers_per_group * self.layer_elements * itemsize
