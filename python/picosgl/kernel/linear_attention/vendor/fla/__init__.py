__version__ = "0.5.2"

from .gated_delta import chunk_gated_delta_rule
from .fused_recurrent import fused_recurrent_gated_delta_rule

__all__ = ["chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"]
