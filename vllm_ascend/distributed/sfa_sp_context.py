from dataclasses import dataclass, field
from typing import Optional, Any

import torch
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.forward_context import get_forward_context

@dataclass
class SfaSpContext:
    num_tokens: int = 0
    num_tokens_pad: int = 0
    local_start: int = 0
    local_end: int = 0
    local_end_with_pad: int = 0
    pad_size: int = 0
    local_pad_size: int = 0

_sfa_sp_context: Optional[SfaSpContext] = None

def set_sfa_sp_context_none():
    global _sfa_sp_context
    _sfa_sp_context = None

def set_sfa_sp_context(hidden_states: torch.Tensor, num_actual_tokens: int):
    global _sfa_sp_context
    tp_group = get_tp_group()
    tp_size = tp_group.world_size
    if _sfa_sp_context is None:
        _sfa_sp_context = SfaSpContext()
    forward_context = get_forward_context()
    assert forward_context.sp_enabled
    num_input_tokens = num_actual_tokens
    num_tokens_per_device = hidden_states.shape[0]
    _sfa_sp_context.num_tokens = num_input_tokens
    _sfa_sp_context.num_tokens_pad = tp_size * num_tokens_per_device
    _sfa_sp_context.pad_size = _sfa_sp_context.num_tokens_pad - _sfa_sp_context.num_tokens
    _sfa_sp_context.local_start = get_tp_group().rank_in_group * num_tokens_per_device
    _sfa_sp_context.local_end_with_pad = _sfa_sp_context.local_start + num_tokens_per_device
    _sfa_sp_context.local_end = min(_sfa_sp_context.local_end_with_pad, num_input_tokens)
    _sfa_sp_context.local_pad_size = _sfa_sp_context.local_end_with_pad - _sfa_sp_context.local_end

def get_sfa_sp_context() -> Optional[SfaSpContext]:
    return _sfa_sp_context

def calc_div_ceil(up: int, down: int) -> int:
    return (up + down - 1) // down

def check_diff(a: torch.Tensor, b: torch.Tensor) -> Any:
    if torch.equal(a, b):
        absolute = torch.abs(a - b)
        relative = torch.abs(a - b) / (torch.abs(a) + 1e-9)
        return (torch.max(absolute).item(), torch.max(relative).item())
    return False

