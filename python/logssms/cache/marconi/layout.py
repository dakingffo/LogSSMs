from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class HybridBlockLayout:
    block_size: int
    layers_per_group: int
    state_groups: int
    state_blocks_per_group: int
    layer_elements: int
    kv_heads: int
    head_dim: int
    state_shape: tuple[int, int, int]
    conv_shape: tuple[int, int]

    @classmethod
    def make(cls, attention_layers, linear_spec, kv_heads, head_dim, block_size=128):
        if attention_layers < 1:
            raise ValueError("Marconi requires at least one full-attention layer")
        state_elements = prod(linear_spec.state_shape) + prod(linear_spec.conv_shape)
        token_elements = 2 * kv_heads * head_dim
        minimum = ((state_elements + token_elements * 64 - 1) // (token_elements * 64)) * 64
        if block_size < 0 or block_size % 64:
            raise ValueError("Marconi block size must be zero (auto) or a multiple of 64")
        block_size = block_size or minimum
        layer_elements = block_size * token_elements
        return cls(block_size, attention_layers,
                   (linear_spec.num_layers + attention_layers - 1) // attention_layers,
                   (state_elements + layer_elements - 1) // layer_elements,
                   layer_elements, kv_heads, head_dim,
                   linear_spec.state_shape, linear_spec.conv_shape)

    @property
    def state_blocks(self) -> int:
        return self.state_groups * self.state_blocks_per_group

    def block_bytes(self, itemsize: int) -> int:
        return self.layers_per_group * self.layer_elements * itemsize
