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

"""Detect-time request input filters (include/exclude chain).

``InputFilterManager`` is a process-wide singleton; detectors call
``InputFilterManager.get().allow(...)`` before running checks. Manual
``manual_trigger`` does not consult the manager.

Evaluation order (within include / exclude): ``prompt_length`` then
``input_token_id_prefix`` then ``prompt_contains_token_ids`` so cheap
checks short-circuit. Allow results are cached per ``req_id`` until
``clear_req`` / finished, or the filter list actually changes (identical
configs on refresh are a no-op and keep the cache).

A *filter config* is one JSON object (type/mode/...). ``input_filter_configs``
is the list of such objects after validation — not the runtime Filter instances.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from vllm_ascend.dfx.request_state import RequestDfxStore
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

MODE_INCLUDE = "include"
MODE_EXCLUDE = "exclude"
_VALID_MODES = frozenset({MODE_INCLUDE, MODE_EXCLUDE})

LENGTH_OPS = frozenset({"gt", "gte", "lt", "lte", "eq", "between"})
CONTAINS_MATCH = frozenset({"any", "subsequence"})


@dataclass(frozen=True)
class InputFilterContext:
    """Inputs available when deciding whether to detect for a request."""

    req_id: str
    prompt_token_ids: list[int] | None


class InputFilter(Protocol):
    mode: str

    def matches(self, ctx: InputFilterContext) -> bool:
        """Whether this filter's condition holds (independent of include/exclude)."""


def matches_input_token_id_prefixes(
    token_ids: Sequence[int],
    prefixes: Sequence[Sequence[int]],
) -> bool:
    """True if ``token_ids`` starts with any prefix, or ``prefixes`` is empty."""
    if not prefixes:
        return True
    ids = list(token_ids)
    for prefix in prefixes:
        pref = list(prefix)
        if not pref:
            return True
        n = len(pref)
        if len(ids) >= n and ids[:n] == pref:
            return True
    return False


@dataclass(frozen=True)
class InputTokenIdPrefixFilter:
    """Match when prompt starts with any configured token-id prefix."""

    mode: str
    prefixes: tuple[tuple[int, ...], ...]

    def matches(self, ctx: InputFilterContext) -> bool:
        if ctx.prompt_token_ids is None:
            return False
        if not self.prefixes:
            return True
        return matches_input_token_id_prefixes(ctx.prompt_token_ids, self.prefixes)


@dataclass(frozen=True)
class PromptLengthFilter:
    """Match prompt length with ``gt|gte|lt|lte|eq|between``."""

    mode: str
    op: str
    value: int | None = None
    min: int | None = None
    max: int | None = None

    def matches(self, ctx: InputFilterContext) -> bool:
        if ctx.prompt_token_ids is None:
            return False
        length = len(ctx.prompt_token_ids)
        op = self.op
        if op == "gt":
            return length > int(self.value or 0)
        if op == "gte":
            return length >= int(self.value or 0)
        if op == "lt":
            return length < int(self.value or 0)
        if op == "lte":
            return length <= int(self.value or 0)
        if op == "eq":
            return length == int(self.value or 0)
        if op == "between":
            lo = int(self.min if self.min is not None else 0)
            hi = int(self.max if self.max is not None else lo)
            return lo <= length <= hi
        return False


@dataclass(frozen=True)
class PromptContainsTokenIdsFilter:
    """Match when prompt contains token ids (``any`` token or ``subsequence``)."""

    mode: str
    token_ids: tuple[int, ...]
    match: Literal["any", "subsequence"] = "any"

    def matches(self, ctx: InputFilterContext) -> bool:
        if ctx.prompt_token_ids is None:
            return False
        if not self.token_ids:
            return True
        ids = ctx.prompt_token_ids
        if self.match == "any":
            needle = set(self.token_ids)
            return any(t in needle for t in ids)
        # Contiguous subsequence.
        n = len(self.token_ids)
        if n == 0:
            return True
        if len(ids) < n:
            return False
        pref = list(self.token_ids)
        return any(ids[i : i + n] == pref for i in range(len(ids) - n + 1))


def _filter_eval_cost(filt: InputFilter) -> int:
    """Lower runs first: length (O(1)) → prefix → contains scan."""
    if isinstance(filt, PromptLengthFilter):
        return 0
    if isinstance(filt, InputTokenIdPrefixFilter):
        return 1
    if isinstance(filt, PromptContainsTokenIdsFilter):
        return 2
    return 3


@dataclass(frozen=True)
class InputFilterChain:
    """Include-all AND exclude-none aggregation over input filters.

    Within each mode, filters are ordered by :func:`_filter_eval_cost` so
    ``prompt_length`` short-circuits before prefix / contains.
    """

    filters: tuple[InputFilter, ...]
    _includes: tuple[InputFilter, ...] = field(init=False, repr=False)
    _excludes: tuple[InputFilter, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        includes = tuple(
            sorted(
                (f for f in self.filters if f.mode == MODE_INCLUDE),
                key=_filter_eval_cost,
            )
        )
        excludes = tuple(
            sorted(
                (f for f in self.filters if f.mode == MODE_EXCLUDE),
                key=_filter_eval_cost,
            )
        )
        object.__setattr__(self, "_includes", includes)
        object.__setattr__(self, "_excludes", excludes)

    def allow(self, ctx: InputFilterContext) -> bool:
        if not self.filters:
            return True
        if self._includes and not all(f.matches(ctx) for f in self._includes):
            return False
        return not any(f.matches(ctx) for f in self._excludes)


def _parse_mode(raw: Any, *, index: int) -> str:
    mode = str(raw if raw is not None else MODE_INCLUDE).lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"input_filter.filters[{index}].mode must be '{MODE_INCLUDE}' or '{MODE_EXCLUDE}', got {raw!r}"
        )
    return mode


def _parse_int_list(raw: Any, *, label: str) -> list[int]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{label} must be a list of ints")
    try:
        return [int(x) for x in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} entries must be ints") from exc


def _parse_prefix_lists(raw: Any, *, label: str) -> list[list[int]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list of int lists")
    out: list[list[int]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, (list, tuple)):
            raise ValueError(f"{label}[{i}] must be a list of ints, got {type(item).__name__}")
        try:
            out.append([int(x) for x in item])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{i}] entries must be ints") from exc
    return out


def normalize_input_filter_configs(configs: Any) -> list[dict[str, Any]]:
    """Validate / normalize ``input_filter.filters`` JSON list into filter configs."""
    if configs is None:
        return []
    if not isinstance(configs, list):
        raise ValueError("input_filter.filters must be a list of filter objects")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(configs):
        if not isinstance(item, dict):
            raise ValueError(f"input_filter.filters[{i}] must be an object")
        ftype = str(item.get("type", "")).strip()
        if not ftype:
            raise ValueError(f"input_filter.filters[{i}].type is required")
        mode = _parse_mode(item.get("mode"), index=i)
        normalized: dict[str, Any] = {"type": ftype, "mode": mode}
        if ftype in ("input_token_id_prefix", "prefix"):
            prefixes = _parse_prefix_lists(
                item.get("prefixes", []),
                label=f"input_filter.filters[{i}].prefixes",
            )
            normalized["type"] = "input_token_id_prefix"
            normalized["prefixes"] = prefixes
        elif ftype in ("prompt_length", "length"):
            op = str(item.get("op", "eq")).lower()
            if op not in LENGTH_OPS:
                raise ValueError(f"input_filter.filters[{i}].op must be one of {sorted(LENGTH_OPS)}, got {op!r}")
            normalized["type"] = "prompt_length"
            normalized["op"] = op
            if op == "between":
                if "min" not in item and "max" not in item:
                    raise ValueError(f"input_filter.filters[{i}] between requires min and/or max")
                if "min" in item and item["min"] is not None:
                    normalized["min"] = int(item["min"])
                if "max" in item and item["max"] is not None:
                    normalized["max"] = int(item["max"])
                lo = normalized.get("min", 0)
                hi = normalized.get("max", lo)
                if hi < lo:
                    raise ValueError(f"input_filter.filters[{i}] between max < min")
            else:
                if "value" not in item:
                    raise ValueError(f"input_filter.filters[{i}] op={op} requires value")
                normalized["value"] = int(item["value"])
        elif ftype in ("prompt_contains_token_ids", "contains_token_ids", "contains"):
            token_ids = _parse_int_list(
                item.get("token_ids", []),
                label=f"input_filter.filters[{i}].token_ids",
            )
            match = str(item.get("match", "any")).lower()
            if match not in CONTAINS_MATCH:
                raise ValueError(f"input_filter.filters[{i}].match must be 'any' or 'subsequence', got {match!r}")
            normalized["type"] = "prompt_contains_token_ids"
            normalized["token_ids"] = token_ids
            normalized["match"] = match
        else:
            raise ValueError(
                f"input_filter.filters[{i}].type unsupported: {ftype!r} "
                "(supported: input_token_id_prefix, prompt_length, prompt_contains_token_ids)"
            )
        out.append(normalized)
    return out


def build_input_filter(config: dict[str, Any]) -> InputFilter:
    ftype = config["type"]
    mode = config["mode"]
    if ftype == "input_token_id_prefix":
        prefixes = tuple(tuple(p) for p in config.get("prefixes", []))
        return InputTokenIdPrefixFilter(mode=mode, prefixes=prefixes)
    if ftype == "prompt_length":
        return PromptLengthFilter(
            mode=mode,
            op=str(config["op"]),
            value=config.get("value"),
            min=config.get("min"),
            max=config.get("max"),
        )
    if ftype == "prompt_contains_token_ids":
        return PromptContainsTokenIdsFilter(
            mode=mode,
            token_ids=tuple(int(x) for x in config.get("token_ids", [])),
            match=config.get("match", "any"),  # type: ignore[arg-type]
        )
    raise ValueError(f"unsupported input filter type: {ftype!r}")


def build_input_filter_chain(configs: Sequence[dict[str, Any]]) -> InputFilterChain:
    filters = tuple(build_input_filter(dict(c)) for c in configs)
    return InputFilterChain(filters=filters)


def _prompt_ids_from_req_states(
    req_states: Any,
    req_id: str,
    req_idx: int | None,
) -> list[int] | None:
    """Read prompt ids from MRV2 ``RequestState`` (UVA CPU mirror preferred)."""
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
    # StagedWriteTensor with uva_instead_of_gpu keeps a host mirror in _uva_buf.
    uva_buf = getattr(all_token_ids, "_uva_buf", None)
    row = None
    if uva_buf is not None:
        host = getattr(uva_buf, "np", None)
        if host is None:
            host = getattr(uva_buf, "cpu", None)
        if host is not None:
            try:
                row = host[idx, :n]
            except (IndexError, TypeError):
                row = None
    if row is None:
        gpu = getattr(all_token_ids, "gpu", None)
        if gpu is None:
            return None
        try:
            row = gpu[idx, :n]
            if hasattr(row, "detach"):
                row = row.detach()
            if hasattr(row, "cpu"):
                row = row.cpu()
        except Exception:
            return None
    if hasattr(row, "tolist"):
        return [int(x) for x in row.tolist()]
    return [int(x) for x in row]


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
            # Some v2 paths carry prompt+partial output as prefill_token_ids.
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
    """Best-effort prompt token ids from runner request state / input batch / MRV2."""
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

    # MRV2: persistent request table (filled after prepare_inputs / prior waves).
    from_states = _prompt_ids_from_req_states(getattr(runner, "req_states", None), req_id, req_idx)
    if from_states is not None:
        return from_states

    # MRV2 first prefill wave: sync_for_step runs before prepare_inputs.
    return _prompt_ids_from_scheduler_output(scheduler_output, req_id)


def iter_batch_prompt_token_ids(
    runner: Any,
    scheduler_output: Any | None = None,
) -> list[tuple[str, int, list[int]]]:
    """Collect ``(req_id, req_idx, prompt_token_ids)`` for the current batch.

    Skips requests whose prompt ids are unavailable.
    """
    if runner is None:
        return []
    input_batch = getattr(runner, "input_batch", None)
    req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None
    out: list[tuple[str, int, list[int]]] = []
    if req_ids:
        for idx, req_id in enumerate(req_ids):
            if not req_id:
                continue
            ids = prompt_token_ids_for_request(runner, str(req_id), idx, scheduler_output=scheduler_output)
            if ids is None:
                continue
            out.append((str(req_id), idx, ids))
        return out

    requests = getattr(runner, "requests", None)
    if isinstance(requests, dict) and requests:
        for req_id, req in requests.items():
            if not req_id:
                continue
            ids = prompt_token_ids_for_request(runner, str(req_id), None, scheduler_output=scheduler_output)
            if ids is None:
                continue
            out.append((str(req_id), -1, ids))
        return out

    # MRV2: iterate req_states, then any scheduled_new_reqs not already covered.
    req_states = getattr(runner, "req_states", None)
    id_map = getattr(req_states, "req_id_to_index", None) if req_states is not None else None
    seen: set[str] = set()
    if isinstance(id_map, dict) and id_map:
        for req_id, idx in sorted(id_map.items(), key=lambda item: int(item[1])):
            if not req_id:
                continue
            ids = prompt_token_ids_for_request(runner, str(req_id), int(idx), scheduler_output=scheduler_output)
            if ids is None:
                continue
            seen.add(str(req_id))
            out.append((str(req_id), int(idx), ids))

    if scheduler_output is not None:
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None) or []
        for req in new_reqs:
            req_id = getattr(req, "req_id", None)
            if not req_id or str(req_id) in seen:
                continue
            ids = prompt_token_ids_for_request(runner, str(req_id), None, scheduler_output=scheduler_output)
            if ids is None:
                continue
            out.append((str(req_id), -1, ids))
    return out


class InputFilterManager:
    """Process-wide singleton owning the live detect-time input filter chain."""

    _instance: InputFilterManager | None = None

    def __init__(self) -> None:
        self._chain: InputFilterChain = InputFilterChain(filters=())
        # Last applied normalized configs; unchanged → skip rebuild / cache clear.
        self._applied_configs: list[dict[str, Any]] = []

    @classmethod
    def get(cls) -> InputFilterManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop singleton (unit tests only)."""
        cls._instance = None
        RequestDfxStore.reset_for_tests()

    def apply_configs(self, configs: Sequence[dict[str, Any]] | None) -> bool:
        """Rebuild chain from ``configs``.

        Returns True if the chain was rebuilt (and allow-cache cleared).
        Identical configs are a no-op so per-step ``refresh_config`` keeps
        ``req_id`` allow results.
        """
        if not isinstance(configs, list):
            configs = []
        else:
            configs = list(configs)
        if configs == self._applied_configs:
            return False
        self._chain = build_input_filter_chain(configs)
        self._applied_configs = configs
        RequestDfxStore.get().clear_all_filter_allowed()
        return True

    def apply_from_config(self, dfx_config: Any | None) -> bool:
        """Pull ``input_filter.filters`` from live config; skip if unchanged."""
        configs: list[dict[str, Any]] = []
        if dfx_config is not None:
            getter = getattr(dfx_config, "input_filter_configs", None)
            if callable(getter):
                raw = getter()
                if isinstance(raw, list):
                    configs = raw
        return self.apply_configs(configs)

    def clear_req(self, req_id: str) -> None:
        """Drop cached allow result when a request finishes.

        Prefer :meth:`RequestDfxStore.clear` from ``DfxProcessor._reap_finished_requests`` / Store.clear.
        """
        RequestDfxStore.get().clear_filter_allowed(req_id)

    def clear_reqs(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self.clear_req(req_id)

    def allow(
        self,
        req_id: str,
        *,
        runner: Any | None = None,
        req_idx: int | None = None,
        prompt_token_ids: Sequence[int] | None = None,
        log: bool = True,
    ) -> bool:
        """True if detectors may run for ``req_id``.

        Empty chain → allow all. Missing prompt while filters configured → deny
        (not cached: prompt may appear on a later step). Otherwise cache by
        ``req_id`` until :meth:`clear_req` / Store.clear or filter-chain rebuild.
        """
        if not self._chain.filters:
            return True
        store = RequestDfxStore.get()
        cached = store.get_filter_allowed(req_id)
        if cached is not None:
            return cached
        ids: list[int] | None
        if prompt_token_ids is not None:
            ids = [int(x) for x in prompt_token_ids]
        else:
            ids = prompt_token_ids_for_request(runner, req_id, req_idx)
        if ids is None:
            if log:
                logger.debug(
                    "[DFX filter] skip detect req_id=%s: filters set but prompt_token_ids unavailable",
                    req_id,
                )
            return False
        ok = self._chain.allow(InputFilterContext(req_id=req_id, prompt_token_ids=ids))
        store.set_filter_allowed(req_id, ok)
        if not ok and log:
            logger.debug(
                "[DFX filter] skip detect req_id=%s: input filter reject prompt_len=%d filters=%d",
                req_id,
                len(ids),
                len(self._chain.filters),
            )
        return ok
