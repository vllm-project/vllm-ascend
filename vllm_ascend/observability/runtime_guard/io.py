#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Token helpers + report-time I/O snapshots over RequestGuardStore."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

# ---- token helpers ----


def is_int_list(value: Any) -> bool:
    """True when ``value`` is a non-empty ``list[int]`` (bool excluded)."""
    return (
        isinstance(value, list) and bool(value) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    )


def is_list_of_int_lists(value: Any) -> bool:
    """True when ``value`` is a non-empty list of int lists."""
    return isinstance(value, list) and bool(value) and all(is_int_list(x) for x in value)


def normalize_token_ids(token_ids: Any) -> list[int]:
    """Normalize tensor / nested tensors / sequences to ``list[int]``."""
    if token_ids is None:
        return []
    if torch.is_tensor(token_ids):
        return [int(x) for x in token_ids.tolist()]
    out: list[int] = []
    for token_id in token_ids:
        if isinstance(token_id, torch.Tensor):
            out.append(int(token_id.item()))
        else:
            out.append(int(token_id))
    return out


def filter_valid_token_ids(token_ids: Any) -> list[int]:
    """Normalize and drop async / pad placeholders (``-1``)."""
    return [tid for tid in normalize_token_ids(token_ids) if tid != -1]


def has_nonempty_sampled_row(req_ids: list[str] | None, sampled_rows: Any) -> bool:
    """True if any request row has at least one valid (non-placeholder) token id.

    Used as a cheap gate for whether after-sample CPU detect should run.
    """
    ids = list(req_ids or [])
    if not ids or sampled_rows is None:
        return False
    for i, _rid in enumerate(ids):
        try:
            row = sampled_rows[i]
        except (IndexError, TypeError, KeyError):
            continue
        if filter_valid_token_ids(row):
            return True
    return False


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode a token-id list to text (``skip_special_tokens=False``)."""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def load_model_tokenizer(runner: Any) -> Any | None:
    """Load model tokenizer via ``cached_tokenizer_from_config``.

    Returns ``None`` if runner/config missing; raises if the load itself fails.
    """
    if runner is None:
        return None
    from vllm.tokenizers import cached_tokenizer_from_config

    vllm_config = getattr(runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None) if vllm_config is not None else None
    if model_config is None:
        return None
    return cached_tokenizer_from_config(model_config)


# ---- I/O snapshot ----


def _raw_output_token_ids(runner: Any, req_id: str, req_idx: int | None) -> Any:
    if runner is None:
        return None
    input_batch = getattr(runner, "input_batch", None)
    req_output_token_ids = getattr(input_batch, "req_output_token_ids", None) if input_batch else None
    if req_output_token_ids is not None and req_idx is not None and 0 <= int(req_idx) < len(req_output_token_ids):
        return req_output_token_ids[int(req_idx)]
    requests = getattr(runner, "requests", None)
    req_state = requests.get(req_id) if requests is not None else None
    if req_state is not None:
        return getattr(req_state, "output_token_ids", None)
    return None


def output_token_count_for_request(runner: Any, req_id: str, req_idx: int | None = None) -> int:
    """Length of cumulative output; prefer runtime_guard Store list when present."""
    st = RequestGuardStore.get().get_state(req_id)
    built = len(st.output_token_ids) if st is not None else 0
    if built > 0:
        return built
    raw = _raw_output_token_ids(runner, req_id, req_idx)
    if raw is not None:
        try:
            ids = normalize_token_ids(raw)
        except TypeError:
            ids = []
        count = sum(1 for tid in ids if tid != -1)
        if count > 0:
            return count
    v2 = _output_from_req_states(runner, req_id, req_idx)
    if v2 is not None:
        return v2[0]
    return 0


def prompt_token_count_for_request(
    runner: Any,
    req_id: str,
    req_idx: int | None = None,
    scheduler_output: Any | None = None,
) -> int:
    ids = prompt_token_ids_for_request(runner, req_id, req_idx, scheduler_output=scheduler_output)
    return len(ids) if ids is not None else 0


@dataclass(frozen=True)
class RequestIoSnapshot:
    """Prompt / cumulative-output view for one request at report time."""

    req_id: str
    prompt_token_count: int
    output_token_count: int
    prompt_token_ids: list[int] | None = None
    output_token_ids: list[int] | None = None

    def as_detail_fields(self) -> dict[str, Any]:
        """Fields merged into anomaly report detail (counts always; ids optional)."""
        out: dict[str, Any] = {
            "prompt_token_count": self.prompt_token_count,
            "output_token_count": self.output_token_count,
        }
        if self.prompt_token_ids is not None:
            out["prompt_token_ids"] = self.prompt_token_ids
        if self.output_token_ids is not None:
            out["output_token_ids"] = self.output_token_ids
        return out


class RequestIoSnapshotManager:
    """Report I/O view helper (not a second per-req state owner).

    - Accumulates via Store (``append_*`` = normalize + ``RequestGuardStore``).
    - Builds :class:`RequestIoSnapshot` for anomaly reports.
    - Keeps a same-wave snapshot cache only; cleared each ``clear_wave_cache``
      and on Store.clear via registered hook.
    """

    _instance: RequestIoSnapshotManager | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._cache: dict[str, RequestIoSnapshot] = {}
        RequestGuardStore.get().register_on_clear(self.clear_req_cache)

    @classmethod
    def get(cls) -> RequestIoSnapshotManager:
        # B11: double-checked lock — first get() from two threads must not
        # build two managers (and register the hook twice).
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    return cls._instance
        # Store may have been reset_for_tests while this singleton survived.
        RequestGuardStore.get().register_on_clear(cls._instance.clear_req_cache)
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None
        RequestGuardStore.reset_for_tests()

    def clear_wave_cache(self) -> None:
        """Drop snapshot cache and same-wave append dedupe on Store.

        Called at the start of each ``sync_for_step`` / ``refresh_config`` wave
        so content-identical chunks from a later step are not swallowed.
        """
        self._cache.clear()
        RequestGuardStore.get().clear_wave_append_frontier()

    def clear_req_cache(self, req_id: str) -> None:
        """Drop wave snapshot cache entries for ``req_id`` (Store on_clear hook)."""
        if not req_id:
            return
        # Keys are only ``{req_id}|0`` / ``{req_id}|1`` (see snapshot).
        self._cache.pop(f"{req_id}|0", None)
        self._cache.pop(f"{req_id}|1", None)

    def cumulative_output_since(self, req_id: str, consumed: int) -> tuple[int, list[int]]:
        """Tail-only cumulative view: ``(total_len, ids[consumed:])``.

        Avoids a full list copy for detectors that only need new ids since a
        per-req cursor.
        """
        return RequestGuardStore.get().new_output_ids_since(req_id, consumed)

    def append_output(self, req_id: str, token_ids: Any) -> None:
        """Normalize accepted ids and append into :class:`RequestGuardStore`.

        Drops ``-1`` placeholders. Within one engine wave, duplicate chunks
        for the same req are skipped (Store ``last_append_chunk``). Across
        waves, identical chunks are kept after :meth:`clear_wave_cache`.
        """
        if not req_id:
            return
        new_ids = filter_valid_token_ids(token_ids)
        if not new_ids:
            return
        RequestGuardStore.get().append_output_ids(req_id, new_ids)
        # Same-wave ``include_token_ids`` snapshots must not keep pre-append ids.
        self.clear_req_cache(req_id)

    def append_batch(
        self,
        req_ids: list[str] | None,
        sampled_rows: Any,
    ) -> None:
        """Append one step of per-req sampled / accepted token rows."""
        if not req_ids or sampled_rows is None:
            return
        for i, req_id in enumerate(req_ids):
            if not req_id:
                continue
            try:
                row = sampled_rows[i]
            except (IndexError, TypeError, KeyError):
                continue
            self.append_output(req_id, row)

    def snapshot(
        self,
        runner: Any,
        req_id: str,
        req_idx: int | None = None,
        *,
        include_token_ids: bool = False,
        use_cache: bool = True,
        scheduler_output: Any | None = None,
    ) -> RequestIoSnapshot:
        """Build prompt + cumulative-output view for ``req_id``.

        Output prefers Store cumulative ids (async-safe). Prompt is read from
        the runner (MRV1 batch / MRV2 ``req_states`` / ``scheduler_output``).
        ``include_token_ids`` attaches full id lists when
        ``report.save_sensitive_info`` is on.
        """
        if not req_id:
            return RequestIoSnapshot(req_id="", prompt_token_count=0, output_token_count=0)
        # S16 fix: only cache the expensive path (include_token_ids=True).
        # The count-only path is cheap to recompute; caching it returned
        # stale output_token_count when a later append landed in the same
        # wave after the first snapshot() call.
        cacheable = use_cache and include_token_ids
        cache_key = f"{req_id}|{int(bool(include_token_ids))}"
        if cacheable and cache_key in self._cache:
            return self._cache[cache_key]

        prompt_ids: list[int] | None = None
        output_ids: list[int] | None = None
        store = RequestGuardStore.get()
        st_out = store.get_state(req_id)
        built_output = list(st_out.output_token_ids) if st_out is not None else []
        if include_token_ids and runner is not None:
            raw_prompt = prompt_token_ids_for_request(runner, req_id, req_idx, scheduler_output=scheduler_output)
            prompt_ids = list(raw_prompt) if raw_prompt is not None else []
            prompt_count = len(prompt_ids)
            if built_output:
                output_ids = built_output
                output_count = len(output_ids)
            else:
                output_ids = filter_valid_token_ids(_raw_output_token_ids(runner, req_id, req_idx))
                if output_ids:
                    output_count = len(output_ids)
                else:
                    # MRV2: decode appends are device-side only; host mirrors
                    # lag forever, so ask the staged tensors directly.
                    v2 = _output_from_req_states(runner, req_id, req_idx)
                    output_count = v2[0] if v2 is not None else 0
                    output_ids = v2[1] if v2 is not None else None
        else:
            prompt_count = prompt_token_count_for_request(runner, req_id, req_idx, scheduler_output=scheduler_output)
            output_count = (
                len(built_output) if built_output else output_token_count_for_request(runner, req_id, req_idx)
            )
            output_ids = None

        snap = RequestIoSnapshot(
            req_id=req_id,
            prompt_token_count=prompt_count,
            output_token_count=output_count,
            prompt_token_ids=prompt_ids,
            output_token_ids=output_ids if include_token_ids else None,
        )
        if cacheable:
            self._cache[cache_key] = snap
        return snap

    def merge_into_detail(
        self,
        detail: dict[str, Any] | None,
        snapshot: RequestIoSnapshot,
    ) -> dict[str, Any]:
        """Overlay I/O fields onto detector detail (I/O wins on key clash)."""
        out = dict(detail or {})
        out.update(snapshot.as_detail_fields())
        return out


def _req_state_index(req_states: Any, req_id: str, req_idx: int | None) -> int | None:
    """Resolve a request's row index in MRV2 ``RequestState``."""
    if req_states is None:
        return None
    id_map = getattr(req_states, "req_id_to_index", None)
    idx: int | None = None
    if req_idx is not None:
        try:
            idx_i = int(req_idx)
        except (TypeError, ValueError):
            idx_i = -1
        if idx_i >= 0:
            idx = idx_i
    if idx is None and isinstance(id_map, dict):
        mapped = id_map.get(req_id)
        if mapped is not None:
            try:
                idx = int(mapped)
            except (TypeError, ValueError):
                return None
    if idx is None or idx < 0:
        return None
    return idx


def _read_staged_row(staged: Any, idx: int, n: int) -> list[int] | None:
    """Read ``row[idx, :n]`` from a vLLM staged/UVA tensor.

    1-D staged tensors (``total_len`` & friends, shape ``[max_num_reqs]``) are
    read as the scalar at ``idx``. On NPU the ascend ``UvaBufferWrapper`` patch
    keeps ``_uva_buf.np`` a host mirror that device-side triton appends never
    update (only ``.gpu`` is current), so an all-zero host row must fall
    through to the device view.
    """
    if n <= 0:
        return []

    def _row(buf: Any) -> Any:
        if int(getattr(buf, "ndim", 2)) == 1:
            return buf[idx : idx + 1]
        return buf[idx, :n]

    uva_buf = getattr(staged, "_uva_buf", None)
    if uva_buf is not None:
        host = getattr(uva_buf, "np", None)
        if host is None:
            host = getattr(uva_buf, "cpu", None)
        if host is not None:
            try:
                row = _row(host)
            except (IndexError, TypeError, ValueError):
                row = None
            if row is not None:
                try:
                    row_list = row.tolist() if hasattr(row, "tolist") else list(row)
                except Exception:
                    row_list = None
                if row_list is not None and not all(int(x) == 0 for x in row_list):
                    return [int(x) for x in row_list]
    gpu = getattr(staged, "gpu", None)
    if gpu is None:
        return None
    try:
        row = _row(gpu)
        if hasattr(row, "detach"):
            row = row.detach()
        if hasattr(row, "cpu"):
            row = row.cpu()
        row_list = row.tolist() if hasattr(row, "tolist") else list(row)
    except Exception:
        return None
    if all(int(x) == 0 for x in row_list):
        return None
    return [int(x) for x in row_list]


def _prompt_ids_from_req_states(
    req_states: Any,
    req_id: str,
    req_idx: int | None,
) -> list[int] | None:
    """Read prompt ids from MRV2 ``RequestState`` (host mirror, device fallback)."""
    idx = _req_state_index(req_states, req_id, req_idx)
    if idx is None:
        return None

    prompt_len = getattr(req_states, "prompt_len", None)
    prompt_len_np = getattr(prompt_len, "np", None) if prompt_len is not None else None
    if prompt_len_np is None:
        return None
    try:
        n = int(prompt_len_np[idx])
    except (IndexError, TypeError, ValueError):
        return None
    if n <= 0:
        return []

    all_token_ids = getattr(req_states, "all_token_ids", None)
    if all_token_ids is None:
        return None
    return _read_staged_row(all_token_ids, idx, n)


def _output_from_req_states(
    runner: Any,
    req_id: str,
    req_idx: int | None,
) -> tuple[int, list[int] | None] | None:
    """(count, ids|None) for MRV2 requests: ``total_len - prompt_len``.

    Decode appends on MRV2 are device-side (triton kernel writes
    ``all_token_ids.gpu`` / ``total_len.gpu``), so the device view is the only
    current source; the host mirror lags forever.
    """
    req_states = getattr(runner, "req_states", None)
    idx = _req_state_index(req_states, req_id, req_idx)
    if idx is None:
        return None
    prompt_len = getattr(req_states, "prompt_len", None)
    prompt_len_np = getattr(prompt_len, "np", None) if prompt_len is not None else None
    if prompt_len_np is None:
        return None
    try:
        prompt_n = int(prompt_len_np[idx])
    except (IndexError, TypeError, ValueError):
        return None
    total_len = getattr(req_states, "total_len", None)
    if total_len is None:
        return None
    total_row = _read_staged_row(total_len, idx, 1)
    if not total_row:
        return None
    total_n = int(total_row[0])
    out_n = max(0, total_n - prompt_n)
    if out_n == 0:
        return 0, []
    all_token_ids = getattr(req_states, "all_token_ids", None)
    if all_token_ids is None:
        return out_n, None
    ids = _read_staged_row(all_token_ids, idx, prompt_n + out_n)
    if ids is None:
        return out_n, None
    return out_n, ids[prompt_n:]


def _prompt_ids_from_scheduler_output(
    scheduler_output: Any,
    req_id: str,
) -> list[int] | None:
    """First-wave MRV2: prompts live on ``scheduled_new_reqs`` before prepare_inputs."""
    if scheduler_output is None or not req_id:
        return None
    new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    if not new_reqs:
        return None
    for req in new_reqs:
        if getattr(req, "req_id", None) != req_id:
            continue
        ids = getattr(req, "prompt_token_ids", None)
        if ids is None:
            ids = getattr(req, "prefill_token_ids", None)
        if ids is None:
            return None
        return [int(x) for x in ids]
    return None


def prompt_token_ids_for_request(
    runner: Any,
    req_id: str,
    req_idx: int | None = None,
    scheduler_output: Any | None = None,
) -> list[int] | None:
    """Best-effort prompt token ids from Store / runner request state / MRV2."""
    st = RequestGuardStore.get().get_state(req_id)
    if st is not None and st.prompt_token_ids is not None:
        return list(st.prompt_token_ids)
    if runner is None:
        return None

    requests = getattr(runner, "requests", None)
    if isinstance(requests, dict):
        req = requests.get(req_id)
        if req is not None:
            ids = getattr(req, "prompt_token_ids", None)
            if ids is not None:
                return [int(x) for x in ids]

    input_batch = getattr(runner, "input_batch", None)
    if input_batch is not None:
        idx = req_idx
        if idx is None:
            req_id_to_index = getattr(input_batch, "req_id_to_index", None)
            if isinstance(req_id_to_index, dict):
                idx = req_id_to_index.get(req_id)
        if idx is not None:
            try:
                idx_i = int(idx)
            except (TypeError, ValueError):
                idx_i = -1
            token_ids_cpu = getattr(input_batch, "token_ids_cpu", None)
            num_prompt_tokens = getattr(input_batch, "num_prompt_tokens", None)
            if (
                token_ids_cpu is not None
                and num_prompt_tokens is not None
                and idx_i >= 0
                and idx_i < len(num_prompt_tokens)
            ):
                n = int(num_prompt_tokens[idx_i])
                if n <= 0:
                    return []
                row = token_ids_cpu[idx_i, :n]
                if hasattr(row, "tolist"):
                    return [int(x) for x in row.tolist()]
                return [int(x) for x in row]

    from_states = _prompt_ids_from_req_states(getattr(runner, "req_states", None), req_id, req_idx)
    if from_states is not None:
        return from_states

    return _prompt_ids_from_scheduler_output(scheduler_output, req_id)
