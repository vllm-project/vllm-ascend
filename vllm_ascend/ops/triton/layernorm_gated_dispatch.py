"""Pure-scalar PR1 launch selection for the gated LayerNorm kernel.

The selector deliberately knows nothing about torch, devices, or resource
queries.  The wrapper supplies the initialized vector-core count on NPU and
``None`` for non-NPU tensors.  A missing count therefore means that the
upstream BASE64 launch must be retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple


class LaunchSpec(NamedTuple):
    impl: Literal["FT_BASE", "FT_PERSIST", "FT_PERSIST_HOIST"]
    block_m: int


BM_PERSIST_SINGLE = 32
# A tile wave is one BM32 tile per initialized vector core.
HOIST_MIN_TILE_WAVES = 16


@dataclass(frozen=True)
class DispatchParams:
    """M-axis routing policy for the per-group width ``N_group``.

    ``bm_*`` are row-tile heights. ``k_persist_num/den`` is the minimum
    BM32 tile count per vector core for persistent execution; ``None`` means
    that a route has not been configured and is rejected if reached.
    """

    bm_small: int | None = None
    bm_multi: int | None = None
    k_persist_num: int | None = None
    k_persist_den: int | None = None
    n_persist_min: int = 1
    hoist_qualified: bool = False
    persist_single_qualified: bool = False


DEFAULT_PARAMS = DispatchParams()


class DispatchConfigError(ValueError):
    """Raised when a selector input or policy is not materialized."""


def _is_positive_exact_int(value) -> bool:
    return type(value) is int and value > 0


def _ceil_div(value: int, divisor: int) -> int:
    if not _is_positive_exact_int(value) or not _is_positive_exact_int(divisor):
        raise DispatchConfigError("ceil-div inputs must be positive exact ints")
    return (value + divisor - 1) // divisor


def _need(value, name: str):
    if value is None:
        raise DispatchConfigError(f"{name} not materialized")
    return value


def validate_params(params) -> None:
    if type(params) is not DispatchParams:
        raise DispatchConfigError("params must be a DispatchParams instance")
    for name in ("hoist_qualified", "persist_single_qualified"):
        if type(getattr(params, name)) is not bool:
            raise DispatchConfigError(f"{name} must be an exact bool")
    if not _is_positive_exact_int(params.n_persist_min):
        raise DispatchConfigError("n_persist_min must be a positive exact int")
    for name in ("bm_small", "bm_multi", "k_persist_num", "k_persist_den"):
        value = getattr(params, name)
        if value is not None and not _is_positive_exact_int(value):
            raise DispatchConfigError(f"{name} must be None or a positive exact int")
    if params.hoist_qualified and not params.persist_single_qualified:
        raise DispatchConfigError("hoist_qualified implies persist_single_qualified")


def _validate_inputs(M, N_group, ngroups, runtime_p) -> None:
    for name, value in (("M", M), ("N_group", N_group), ("ngroups", ngroups)):
        if not _is_positive_exact_int(value):
            raise DispatchConfigError(f"{name} must be a positive exact int")
    if runtime_p is not None and not _is_positive_exact_int(runtime_p):
        raise DispatchConfigError("runtime_p must be None or a positive exact int")


def _select_layernorm_launch(
    M,
    N_group,
    ngroups,
    runtime_p,
    params: DispatchParams = DEFAULT_PARAMS,
) -> LaunchSpec:
    """Select the bounded PR1 path, or the upstream BASE64 fallback.

    A missing ``runtime_p`` selects the upstream BASE64 launch.  NPU callers
    obtain it through the existing initialized-device-properties contract.
    """
    validate_params(params)
    _validate_inputs(M, N_group, ngroups, runtime_p)
    if runtime_p is None:
        return LaunchSpec("FT_BASE", 64)

    # PR1 does not own a wide-N path.  The wrapper performs the upstream
    # BASE64 launch for this domain; keeping this result scalar makes the
    # fallback explicit in selector tests too.
    if N_group > 128:
        return LaunchSpec("FT_BASE", 64)

    if N_group < _need(params.n_persist_min, "n_persist_min"):
        return LaunchSpec("FT_BASE", _need(params.bm_small, "bm_small"))

    bm_persist = BM_PERSIST_SINGLE if ngroups == 1 else _need(params.bm_multi, "bm_multi")

    # The only qualified multi-group PR1 route is BASE32 at N_group=128.
    if N_group == 128 and ngroups > 1:
        return LaunchSpec("FT_BASE", bm_persist)

    # A persistent launch is considered once its tile count reaches a
    # calibrated fraction of the initialized vector-core count.
    persist_tiles = _ceil_div(M, bm_persist) * ngroups
    if (
        persist_tiles * _need(params.k_persist_den, "k_persist_den")
        < _need(params.k_persist_num, "k_persist_num") * runtime_p
    ):
        return LaunchSpec("FT_BASE", _need(params.bm_small, "bm_small"))

    if ngroups == 1:
        if params.hoist_qualified and persist_tiles >= HOIST_MIN_TILE_WAVES * runtime_p:
            return LaunchSpec("FT_PERSIST_HOIST", BM_PERSIST_SINGLE)
        if params.persist_single_qualified:
            return LaunchSpec("FT_PERSIST", BM_PERSIST_SINGLE)
        return LaunchSpec("FT_BASE", BM_PERSIST_SINGLE)

    return LaunchSpec("FT_BASE", _need(params.bm_multi, "bm_multi"))
