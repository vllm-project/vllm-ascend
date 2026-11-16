#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Per-head value amax calibration for the QFA INT8 static ``fa_v.scale``.

Arms itself only when ``VLLM_ASCEND_CALIB_FA_V`` points to a writable
directory (see envs.py). Each worker process wraps ``Attention.forward``
and keeps, per layer, the running per-head amax of the value tensor: the
value entering ``Attention.forward`` is exactly what ``reshape_and_cache``
writes into the KV cache (V has no RoPE), so it is the right calibration
signal for a static per-head scale (amax / 127, matching
``QFA_INT8_SCALE_LEVEL`` in attention_a3_qfa.py).

Accumulation stays on device (one small kernel per attention call, no host
sync). One dump per forward step: layers fire in a stable order within a
step, so the first-seen layer name marks the step boundary. A periodic
fallback and an atexit dump cover teardown paths that skip the boundary.

Eager mode only: piecewise/full graphs replay without running python
wrappers, so compiled steps would silently under-count. Drive this from
an offline eager LLM run (fa_v_scale_calib.py), never from a compiled
serving process.
"""

import atexit
import os

import torch

_PERIODIC_DUMP_CALLS = 256


class _CalibState:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self.layers: dict[str, dict] = {}
        self.step_anchor: str | None = None
        self.calls = 0


_state: _CalibState | None = None
_original_forward = None
_failure_logged = False


def _log_failure_once(exc: BaseException) -> None:
    global _failure_logged
    if not _failure_logged:
        _failure_logged = True
        from vllm_ascend.utils import logger

        logger.warning(
            "fa_v_calib: stats update failed once (%s); serving is unaffected", exc
        )


def _rank_and_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _dump() -> None:
    assert _state is not None
    rank, world = _rank_and_world()
    layers = {
        name: {
            "amax": entry["amax"].float().cpu(),
            "calls": entry["calls"],
            "tokens": entry["tokens"],
        }
        for name, entry in _state.layers.items()
    }
    payload = {"rank": rank, "world_size": world, "layers": layers}
    path = os.path.join(_state.out_dir, f"fa_v_stats_rank{rank}.pt")
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _record(module, value: torch.Tensor) -> None:
    name = getattr(module, "layer_name", None) or f"attn_{id(module):x}"
    amax = value.detach().abs().amax(dim=(0, 2)).float()
    entry = _state.layers.get(name)
    if entry is None:
        _state.layers[name] = {"amax": amax, "calls": 1, "tokens": value.shape[0]}
        if _state.step_anchor is None:
            _state.step_anchor = name
        return
    torch.maximum(entry["amax"], amax, out=entry["amax"])
    entry["calls"] += 1
    entry["tokens"] += value.shape[0]
    _state.calls += 1
    if name == _state.step_anchor and _state.calls > 1:
        _dump()
    elif _state.calls % _PERIODIC_DUMP_CALLS == 0:
        _dump()


def _wrapped_forward(self, *args, **kwargs):
    try:
        value = args[2] if len(args) >= 3 else kwargs.get("value")
        if isinstance(value, torch.Tensor) and value.dim() == 2:
            value = value.view(value.shape[0], self.num_kv_heads, -1)
        if (
            isinstance(value, torch.Tensor)
            and value.dim() == 3
            and value.shape[0] > 0
            and value.shape[1] > 0
        ):
            _record(self, value)
    except Exception as exc:  # noqa: BLE001
        _log_failure_once(exc)
    return _original_forward(self, *args, **kwargs)


def maybe_install() -> None:
    """Idempotent; returns immediately when the env var is unset."""
    global _state, _original_forward
    out_dir = os.getenv("VLLM_ASCEND_CALIB_FA_V", "")
    if not out_dir or _state is not None:
        return
    try:
        from vllm.model_executor.layers.attention.attention import Attention
    except ImportError:
        from vllm.model_executor.layers.attention import Attention

    os.makedirs(out_dir, exist_ok=True)
    _state = _CalibState(out_dir)
    _original_forward = Attention.forward
    Attention.forward = _wrapped_forward
    atexit.register(_final_dump)

    from vllm_ascend.utils import logger

    rank, world = _rank_and_world()
    logger.info(
        "fa_v_calib: recording per-layer per-head V amax to %s (rank=%d, world=%d)",
        out_dir,
        rank,
        world,
    )


def _final_dump() -> None:
    if _state is not None and _state.layers:
        try:
            _dump()
        except Exception as exc:  # noqa: BLE001
            _log_failure_once(exc)
