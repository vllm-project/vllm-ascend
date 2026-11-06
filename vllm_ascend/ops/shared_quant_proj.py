#
# Projection of one input through several Linear layers with a single
# quantize - and, under sequence parallelism, with the all-gather moved
# behind that quantize so the collective carries the quantized activation.
#
import torch
from torch import nn
from vllm.distributed import tensor_model_parallel_all_gather

from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather
from vllm_ascend.ops.cv_linear import CVLinearWrapper

_SHARED_QUANT_CACHE = "_ascend_shared_quant_wrappers"


def _shared_quant_signature(wrapper: CVLinearWrapper) -> tuple | None:
    """Identity of what ``wrapper.quantize`` produces, or None if unshareable.

    Two wrappers with the same non-None signature turn the same input into the
    same ``(quantized_x, pertoken_scale)`` pair, so one quantize can feed both
    ``matmul`` calls. A single wrapper with a non-None signature is one whose
    quantize step can be separated from its matmul at all. None means the
    quantize step is a no-op (communication / unquantized paths) or
    ``CVLinearWrapper`` does not model the scheme exactly.
    """
    if wrapper._has_communication:
        return None
    if wrapper._is_w8a8_dynamic:
        # _is_w8a8_dynamic is an isinstance check, so it also matches the
        # W8A8FP8_DYNAMIC subclass, which overrides act_quant_type to fp8.
        # CVLinearWrapper.quantize hardcodes int8 npu_dynamic_quant, so only
        # the int8 scheme is reproduced faithfully here; anything else must
        # keep going through its own quant_method.apply.
        quant_method = wrapper._quant_method
        act_quant_type = getattr(quant_method, "act_quant_type", None)
        if act_quant_type is None:
            inner = getattr(quant_method, "quant_method", None)
            act_quant_type = getattr(inner, "act_quant_type", None)
        if act_quant_type is not torch.int8:
            return None
        return ("w8a8_dynamic",)
    if wrapper._is_mxfp8_dynamic:
        # group_size is not an input to npu_dynamic_mx_quant, so it does not
        # change the quantized tensor; each matmul applies its own.
        return ("mxfp8_dynamic", wrapper._mxfp8_quant_method.dynamic_mx_quant_scale_alg)
    return None


def _build_shared_quant_wrappers(owner: nn.Module, names: tuple[str, ...]) -> list[CVLinearWrapper] | None:
    """Wrappers for ``names`` if they can share one quantize, else None.

    Sharing is only valid when every projection quantizes its input the same
    way (same scheme, same scale algorithm) and none of them folds a bias into
    the matmul, since ``matmul`` is called here without one.
    """
    wrappers = [CVLinearWrapper(getattr(owner, name)) for name in names]
    if any(getattr(w.linear, "bias", None) is not None for w in wrappers):
        return None
    signatures = {_shared_quant_signature(w) for w in wrappers}
    if len(signatures) != 1 or None in signatures:
        return None
    return wrappers


def _get_shared_quant_wrappers(owner: nn.Module, names: tuple[str, ...]) -> list[CVLinearWrapper] | None:
    """Cached ``_build_shared_quant_wrappers`` result for ``names``."""
    cached = getattr(owner, _SHARED_QUANT_CACHE, None)
    if cached is None:
        cached = {}
        setattr(owner, _SHARED_QUANT_CACHE, cached)
    if names not in cached:
        cached[names] = _build_shared_quant_wrappers(owner, names)
    return cached[names]


def _sp_gather_quantized(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather an already-quantized token shard and its scale.

    Dynamic activation quantization is per token row, so quantizing the shard
    and gathering the result is numerically identical to gathering first and
    quantizing the full sequence - but the collective moves one byte per
    element instead of two, and each rank only quantizes its own shard. The
    same ordering the MoE EP path uses in ``prepare_finalize``.

    Trailing rows are ``sp_shard`` zero padding, whose quantization carries no
    information (an all-zero row yields a zero scale); the slice back to
    ``num_tokens`` drops them before any matmul sees them. The plain TP
    collective is used rather than ``sp_all_gather``, whose optional fused
    hook is written for bf16 hidden states, not an int8/fp8 payload and its
    per-token (1-D) or per-group (2-D) scale.
    """
    quantized = tensor_model_parallel_all_gather(quantized, 0)[:num_tokens]
    scale = tensor_model_parallel_all_gather(scale, 0)[:num_tokens]
    return quantized, scale


def project_with_shared_quant(
    owner: nn.Module,
    hidden_states: torch.Tensor,
    names: tuple[str, ...],
    sp_num_tokens: int | None = None,
) -> list[torch.Tensor]:
    """Project ``hidden_states`` through ``names``, quantizing it only once.

    ``sp_num_tokens`` is the full sequence's token count. Passing it means
    ``hidden_states`` holds only this rank's sequence-parallel shard, which
    this function all-gathers back to that many rows: after the quantize when
    the scheme can be split into quantize + matmul, before the per-projection
    forwards otherwise.
    """
    # Splitting quantize from matmul only pays for itself when it removes a
    # redundant quantize (several projections over one input) or lets the
    # collective carry the quantized activation. A lone projection with
    # neither keeps its own scheme's forward.
    if len(names) == 1 and sp_num_tokens is None:
        return [getattr(owner, names[0])(hidden_states)[0]]

    wrappers = _get_shared_quant_wrappers(owner, names)

    if wrappers is None:
        # No shared quantization to move the collective behind.
        if sp_num_tokens is not None:
            hidden_states = sp_all_gather(hidden_states)[:sp_num_tokens]
        return [getattr(owner, name)(hidden_states)[0] for name in names]

    quantized, scale = wrappers[0].quantize(hidden_states)
    if sp_num_tokens is not None:
        # A shared signature implies a quantizing, non-communicating scheme.
        assert scale is not None
        quantized, scale = _sp_gather_quantized(quantized, scale, sp_num_tokens)
    return [w.matmul(quantized, scale) for w in wrappers]
