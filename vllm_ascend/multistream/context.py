from contextlib import contextmanager
from typing import Any

_ms_comm_context: Any = None
_cur_micro_batch_num: int = -1
_ms_layer_index_context: int = -1
_ms_metadata_context: Any = None
_ms_attn_metadata_context: Any = None


def set_multistream_layer_context(start_layer: int, ms_metadata: Any,
                                  attn_metadata: Any):
    """
    set multistream layer context before transformer layers
    """
    global _ms_layer_index_context, _ms_metadata_context, _ms_attn_metadata_context
    _ms_layer_index_context = start_layer
    _ms_metadata_context = ms_metadata
    _ms_attn_metadata_context = attn_metadata


def reset_multistream_layer_context():
    """
    reset multistream layer context
    """
    global _ms_layer_index_context, _ms_metadata_context, _ms_attn_metadata_context
    _ms_layer_index_context = -1
    _ms_metadata_context = None
    _ms_attn_metadata_context = None


def get_multistream_layer_context():
    """
    get multistream layer context
    """
    return _ms_layer_index_context, _ms_metadata_context, _ms_attn_metadata_context


def advance_step_multistream_layer_context():
    """
    advance multistream layer index context
    """
    global _ms_layer_index_context
    _ms_layer_index_context += 1


def get_multistream_comm_context() -> Any:
    """Get the current comm forward context."""
    return _ms_comm_context


def get_multistream_microbatch_context() -> int:
    return _cur_micro_batch_num


@contextmanager
def set_multistream_context(context: Any, micro_batch_num: int):
    """A context manager that stores the current comm forward context,
    can be attention metadata, etc."""
    global _ms_comm_context, _cur_micro_batch_num
    _ms_comm_context = context
    _cur_micro_batch_num = micro_batch_num
    try:
        yield
    finally:
        _ms_comm_context = None
        _cur_micro_batch_num = -1
