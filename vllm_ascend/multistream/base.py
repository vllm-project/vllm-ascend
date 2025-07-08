from dataclasses import dataclass
from enum import Enum


class MSEventKey(Enum):
    ATTN_COM_FINISH = 0
    ATTN_AR_FINISH = 1
    FFN_COM_FINISH = 2
    FFN_AR_FINISH = 3
    # events for MOE dispatch and combine
    MOE_BEFORE_COMM = 4
    MOE_AFTER_COMM = 5
    # events for shared expert
    MOE_SE_COMM_FINISH = 6
    MOE_SE_COMP_FINISH = 7
    MOE_GATE_FINISH = 8
    MOE_ALL_TO_ALL = 9
    MOE_ALL_TO_ALL_FINISH = 10


@dataclass
class MSAttentionMetadataSplitConfig:
    """
    micro batch split config for split attention metadata
    """
    # micro batch num
    num_micro_batches: int = 2
    # split micro batches only when total tokens >= min_total_tokens_to_split
    min_total_tokens_to_split: int = 256
    # split micro batches only when prefill tokens >= min_prefill_tokens_to_split
    min_prefill_tokens_to_split: int = 64
