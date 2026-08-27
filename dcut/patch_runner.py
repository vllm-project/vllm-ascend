# SPDX-License-Identifier: Apache-2.0
"""Patch NPUModelRunner for D-Cut adaptive verify."""
from __future__ import annotations

import os

import torch

from vllm.distributed import get_pp_group, get_tp_group, get_world_group

from .controller import _dcut_init_controller, _dcut_enable_drafter_probs
from .dcut_profile import _adaptive_profile_run
from .draft_profile import _adaptive_profile_draft_run
from .gdn_buffers import (
    _dcut_prepare_gdn_eager_state,
    _dcut_prepare_gdn_graph_capture,
    _dcut_prepare_gdn_piecewise_replay,
)
from .globals import ENV_DISABLE, ENV_FULL_DECODE_ONLY, logger
from .patch_gdn_v023 import _dcut_gdn_has_prefill
from .patch_full_graph import _dcut_full_decode_multishape_enabled
from .patch_piecewise import _is_enabled as _gdn_piecewise_graph_enabled
from .probs import (
    _dcut_bypass_prob_capture_for_prefill,
    _dcut_prepare_prob_capture,
    _dcut_queue_probs,
    _maybe_process_adaptive_probs,
    profile_adaptive_cost,
)
from .truncate import (
    _dcut_add_zero_draft_handoffs,
    _dcut_apply_zero_prob_recompute_caps,
    _dcut_has_prefill,
    _dcut_is_recompute_handoff,
    _dcut_recompute_placeholder_req_ids,
    _dcut_truncate,
    _dcut_zero_draft_kv_handoff_req_ids,
)
from .utils import _env_flag

ENV_DEBUG_STATS = "VLLM_DCUT_DEBUG_STATS"
DUMMY_FIA_KV_SEQ_LEN = 1


def _dcut_ragged_full_fia_rows(
    actual_tokens: int,
    num_tokens_padded: int,
    num_reqs: int,
    request_capacity: int,
) -> int:
    """Return live FIA rows: real requests plus one token-padding request."""
    actual_tokens = int(actual_tokens)
    num_tokens_padded = int(num_tokens_padded)
    num_reqs = int(num_reqs)
    request_capacity = int(request_capacity)
    if num_reqs > request_capacity:
        raise RuntimeError(
            "D-Cut ragged FULL request count exceeds graph capacity: "
            f"requests={num_reqs}, capacity={request_capacity}"
        )
    if actual_tokens > num_tokens_padded:
        raise RuntimeError(
            "D-Cut ragged FULL query length exceeds token bucket: "
            f"actual={actual_tokens}, padded={num_tokens_padded}"
        )
    return num_reqs + int(actual_tokens < num_tokens_padded)


def _dcut_pad_fia_dummy_seq_len(
    metadata,
    num_reqs: int,
    fia_rows: int,
) -> None:
    """Give the optional FIA token-padding request a valid dummy KV length."""
    dummy_start = int(num_reqs)
    dummy_stop = int(fia_rows)
    if dummy_stop not in (dummy_start, dummy_start + 1):
        raise RuntimeError(
            "D-Cut ragged FULL FIA must expose only live requests and at "
            f"most one padding request: requests={dummy_start}, rows={dummy_stop}"
        )
    if dummy_start == dummy_stop:
        return

    seq_lens_list = getattr(metadata, "seq_lens_list", None)
    if seq_lens_list is None or len(seq_lens_list) < dummy_stop:
        length = None if seq_lens_list is None else len(seq_lens_list)
        raise RuntimeError(
            "D-Cut ragged FULL FIA seq-lens list is shorter than its live "
            f"request rows: length={length}, rows={dummy_stop}"
        )
    seq_lens_list[dummy_start:dummy_stop] = [DUMMY_FIA_KV_SEQ_LEN]

    updated_tensors = set()
    for field in ("seq_lens", "seq_lens_cpu"):
        seq_lens = getattr(metadata, field, None)
        if seq_lens is None or id(seq_lens) in updated_tensors:
            continue
        if int(seq_lens.shape[0]) < dummy_stop:
            raise RuntimeError(
                "D-Cut ragged FULL FIA seq-lens tensor is shorter than its "
                f"live request rows: field={field}, "
                f"shape={tuple(seq_lens.shape)}, rows={dummy_stop}"
            )
        seq_lens[dummy_start:dummy_stop].fill_(DUMMY_FIA_KV_SEQ_LEN)
        updated_tensors.add(id(seq_lens))


def _dcut_pad_fixed_drafter_fia_seq_lens(
    metadata,
    num_reqs: int,
    request_capacity: int,
) -> None:
    """Keep the uniform drafter's fixed FIA request axis valid."""
    inactive_start = int(num_reqs)
    # The fixed axis contains request_capacity rows plus the final FIA-only
    # token-padding row. Every zero-query row still needs a positive KV len.
    inactive_stop = int(request_capacity) + 1
    if inactive_start >= inactive_stop:
        return

    seq_lens_list = getattr(metadata, "seq_lens_list", None)
    if seq_lens_list is None or len(seq_lens_list) < inactive_stop:
        length = None if seq_lens_list is None else len(seq_lens_list)
        raise RuntimeError(
            "D-Cut FULL drafter FIA seq-lens list is shorter than its "
            f"request capacity: length={length}, capacity={inactive_stop}"
        )
    seq_lens_list[inactive_start:inactive_stop] = [
        DUMMY_FIA_KV_SEQ_LEN
    ] * (inactive_stop - inactive_start)

    updated_tensors = set()
    for field in ("seq_lens", "seq_lens_cpu"):
        seq_lens = getattr(metadata, field, None)
        if seq_lens is None or id(seq_lens) in updated_tensors:
            continue
        if int(seq_lens.shape[0]) < inactive_stop:
            raise RuntimeError(
                "D-Cut FULL drafter FIA seq-lens tensor is shorter than its "
                f"request capacity: field={field}, "
                f"shape={tuple(seq_lens.shape)}, capacity={inactive_stop}"
            )
        seq_lens[inactive_start:inactive_stop].fill_(
            DUMMY_FIA_KV_SEQ_LEN
        )
        updated_tensors.add(id(seq_lens))


def _dcut_debug_rank_info() -> dict[str, int | bool]:
    """Return stable distributed identifiers for single-writer debug logs."""
    world_group = get_world_group()
    world_rank = int(world_group.rank)
    return {
        "world_rank": world_rank,
        "tp_rank": int(get_tp_group().rank_in_group),
        "pp_rank": int(get_pp_group().rank_in_group),
        "is_writer": world_rank == 0,
    }


def _dcut_adaptive_handoff_graph_enabled(runner) -> bool:
    """Return whether recompute placeholders may join adaptive graph decode."""
    mode = getattr(
        getattr(runner, "compilation_config", None),
        "cudagraph_mode",
        None,
    )
    mode_name = getattr(mode, "name", mode)
    if mode_name == "PIECEWISE":
        return True
    return bool(
        mode_name == "FULL_DECODE_ONLY"
        and _dcut_full_decode_multishape_enabled(runner.vllm_config)
    )


def _dcut_initial_handoff_spec_rows(runner) -> tuple[int, ...]:
    """Map first-step handoff requests to compact speculative GDN rows."""
    handoff_req_ids = getattr(
        runner,
        "_dcut_zero_draft_handoffs_for_proposal",
        frozenset(),
    )
    if not handoff_req_ids:
        return ()

    draft_lens = getattr(
        getattr(runner, "num_decode_draft_tokens", None),
        "np",
        None,
    )
    input_batch = getattr(runner, "input_batch", None)
    if draft_lens is None or input_batch is None:
        return ()

    num_reqs = int(input_batch.num_reqs)
    compact_spec_row = 0
    rows = []
    for req_index, req_id in enumerate(input_batch.req_ids[:num_reqs]):
        if int(draft_lens[req_index]) < 0:
            continue
        if req_id in handoff_req_ids:
            rows.append(compact_spec_row)
        compact_spec_row += 1
    return tuple(rows)


def _dcut_execute_with_gdn_prefill_route(
    runner,
    execute_model,
    scheduler_output,
    intermediate_tensors,
    has_prefill: bool,
):
    """Expose real prefill to GDN without overriding the outer graph mode."""
    attr = "_dcut_gdn_scheduler_has_prefill"
    had_previous = hasattr(runner, attr)
    previous = getattr(runner, attr, None)
    setattr(runner, attr, bool(has_prefill))
    try:
        return execute_model(
            runner,
            scheduler_output,
            intermediate_tensors,
        )
    finally:
        if had_previous:
            setattr(runner, attr, previous)
        elif hasattr(runner, attr):
            delattr(runner, attr)


def _dcut_execute_native_recompute_handoff(
    runner,
    execute_model,
    scheduler_output,
    intermediate_tensors,
):
    """Run a recompute handoff through stock dispatch and GDN metadata."""
    runner_attr = "_dcut_recompute_handoff_active"
    builder_attr = "_dcut_force_native_gdn_metadata"
    had_runner_attr = hasattr(runner, runner_attr)
    previous_runner_value = getattr(runner, runner_attr, None)
    builder_states = []
    seen_builders = set()
    for attention_groups in getattr(runner, "attn_groups", ()):
        for attention_group in attention_groups:
            get_builder = getattr(
                attention_group,
                "get_metadata_builder",
                None,
            )
            if get_builder is None:
                continue
            builder = get_builder(0)
            if id(builder) in seen_builders:
                continue
            attach = getattr(
                type(builder),
                "_attach_spec_decode_metadata",
                None,
            )
            if not getattr(attach, "_dcut_patched", False):
                continue
            seen_builders.add(id(builder))
            builder_states.append(
                (
                    builder,
                    hasattr(builder, builder_attr),
                    getattr(builder, builder_attr, None),
                )
            )
            setattr(builder, builder_attr, True)

    setattr(runner, runner_attr, True)
    try:
        # The native GDN core and its native metadata must be selected as a
        # pair. Placeholder speculative tokens are not prompt prefill, but the
        # existing boolean route is the runner-to-core native-path signal.
        return _dcut_execute_with_gdn_prefill_route(
            runner,
            execute_model,
            scheduler_output,
            intermediate_tensors,
            True,
        )
    finally:
        for builder, had_attr, previous_value in reversed(builder_states):
            if had_attr:
                setattr(builder, builder_attr, previous_value)
            elif hasattr(builder, builder_attr):
                delattr(builder, builder_attr)
        if had_runner_attr:
            setattr(runner, runner_attr, previous_runner_value)
        elif hasattr(runner, runner_attr):
            delattr(runner, runner_attr)


def _dcut_set_full_gdn_metadata_route(runner, use_native: bool) -> None:
    """Pair each FULL descriptor with the matching GDN metadata contract."""
    seen_builders = set()
    for attention_groups in getattr(runner, "attn_groups", ()):
        for attention_group in attention_groups:
            get_builder = getattr(
                attention_group,
                "get_metadata_builder",
                None,
            )
            if get_builder is None:
                continue
            builder = get_builder(0)
            if id(builder) in seen_builders:
                continue
            attach = getattr(
                type(builder),
                "_attach_spec_decode_metadata",
                None,
            )
            if not getattr(attach, "_dcut_patched", False):
                continue
            seen_builders.add(id(builder))
            builder._dcut_force_native_gdn_metadata = bool(use_native)


def _dcut_is_ragged_full_capture(
    runner,
    cudagraph_runtime_mode,
    uniform_decode,
    is_graph_capturing,
) -> bool:
    """Return whether this dummy run captures a target-only ragged FULL key."""
    runtime_mode_name = getattr(
        cudagraph_runtime_mode,
        "name",
        cudagraph_runtime_mode,
    )
    return bool(
        is_graph_capturing
        and not uniform_decode
        and runtime_mode_name == "FULL"
        and _dcut_full_decode_multishape_enabled(runner.vllm_config)
    )


def _dcut_call_dummy_without_drafter(
    runner,
    dummy_run,
    num_tokens,
    args,
    kwargs,
    skip_drafter,
):
    """Run target dummy capture while suppressing only the drafter call."""
    if not skip_drafter:
        return dummy_run(runner, num_tokens, *args, **kwargs)

    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        return dummy_run(runner, num_tokens, *args, **kwargs)

    runner.drafter = None
    try:
        return dummy_run(runner, num_tokens, *args, **kwargs)
    finally:
        runner.drafter = drafter


def _dcut_piecewise_capture_dummy_enabled(
    runner,
    cudagraph_runtime_mode,
    is_profile: bool = False,
    is_graph_capturing: bool = False,
) -> bool:
    """Return whether this dummy run must capture the spec GDN branch."""
    from vllm.config import CUDAGraphMode

    compilation_config = getattr(runner, "compilation_config", None)
    graph_enabled = bool(
        getattr(
            runner,
            "_dcut_gdn_piecewise_enabled",
            _gdn_piecewise_graph_enabled(),
        )
    )
    return (
        graph_enabled
        and not is_profile
        and is_graph_capturing
        and getattr(runner, "_dcut_in_real_warmup", False)
        and getattr(runner, "pcp_size", 1) == 1
        and getattr(runner, "dcp_size", 1) == 1
        and getattr(compilation_config, "cudagraph_mode", None)
        == CUDAGraphMode.PIECEWISE
        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    )


def _patch_runner() -> None:
    import vllm_ascend.worker.model_runner_v1 as m

    R = m.NPUModelRunner
    if getattr(R, "_dcut_patched", False):
        return

    _orig_init = R.__init__

    def __init__(self, *a, **k):
        _orig_init(self, *a, **k)
        self._dcut_gdn_piecewise_enabled = (
            _gdn_piecewise_graph_enabled()
            and getattr(self, "_has_gdn", False)
            and getattr(
                self.compilation_config.cudagraph_mode,
                "name",
                None,
            )
            == "PIECEWISE"
        )
        try:
            _dcut_init_controller(self)
        except Exception as e:
            logger.error("D-Cut init failed; running vanilla: %s", e)
            self._verify_adaptive_controller = None

    _orig_exec = R.execute_model
    _orig_model_forward = R._model_forward
    _orig_dummy_run = R._dummy_run
    _orig_should_build_dummy = R._should_build_dummy_attn_metadata
    _orig_determine = R._determine_batch_execution_and_padding
    _orig_build_attention_metadata = R._build_attention_metadata
    _orig_pad_query_start_loc_for_fia = R._pad_query_start_loc_for_fia

    def _determine_batch_execution_and_padding(
        self,
        num_tokens,
        num_reqs,
        num_scheduled_tokens_np,
        max_num_scheduled_tokens,
        use_cascade_attn,
        allow_microbatching=False,
        force_eager=False,
        force_uniform_decode=None,
        force_has_lora=None,
        force_num_active_loras=None,
        num_encoder_reqs=0,
    ):
        from vllm.config import CUDAGraphMode

        full_multishape = (
            _dcut_full_decode_multishape_enabled(self.vllm_config)
            and self.compilation_config.cudagraph_mode
            == CUDAGraphMode.FULL_DECODE_ONLY
        )
        if (
            full_multishape
            and getattr(self, "_dcut_ragged_full_capture_dummy", False)
            and num_reqs > 0
        ):
            # Stock mixed dummy construction puts the entire remainder in the
            # final request (for example 128 tokens / 96 requests becomes
            # 95x1 + 1x33). That captures a prefill-shaped FIA branch and then
            # replays it for a pure speculative batch whose query lengths are
            # all <= num_spec + 1. Keep the same request capacity and token
            # count, but capture the representative pure-decode geometry.
            base, remainder = divmod(int(num_tokens), int(num_reqs))
            num_scheduled_tokens_np[:num_reqs] = base
            if remainder:
                num_scheduled_tokens_np[:remainder] += 1
            capture_max_query_len = int(
                num_scheduled_tokens_np[:num_reqs].max()
            )
            if capture_max_query_len > int(self.uniform_decode_query_len):
                raise RuntimeError(
                    "D-Cut ragged FULL capture cannot represent the token "
                    "bucket with speculative query lengths: "
                    f"tokens={num_tokens}, requests={num_reqs}, "
                    f"max_query_len={self.uniform_decode_query_len}"
                )
            logger.warning(
                "D-Cut: ragged FULL capture uses pure-decode dummy "
                "token_bucket=%d request_capacity=%d query_len=[%d,%d] "
                "fia_rows=%d.",
                num_tokens,
                num_reqs,
                int(num_scheduled_tokens_np[:num_reqs].min()),
                capture_max_query_len,
                num_reqs + 1,
            )

        if (
            full_multishape
            and getattr(self, "_dcut_recompute_handoff_active", False)
        ):
            # Stock FULL_DECODE_ONLY only graphs a rectangular decode batch.
            # Do not let D-Cut's added ragged keys capture a recompute handoff.
            expected_query_len = self.uniform_decode_query_len
            computed = self.input_batch.num_computed_tokens_cpu[:num_reqs]
            is_all_decode = bool(num_reqs > 0 and (computed > 0).all())
            stock_uniform_decode = bool(
                (is_all_decode if self.speculative_config else True)
                and max_num_scheduled_tokens == expected_query_len
                and num_tokens == expected_query_len * num_reqs
            )
            if not stock_uniform_decode:
                force_eager = True
        elif full_multishape and force_uniform_decode is None:
            computed = self.input_batch.num_computed_tokens_cpu[:num_reqs]
            is_all_decode = bool(num_reqs > 0 and (computed > 0).all())
            draft_lens = getattr(
                getattr(self, "num_decode_draft_tokens", None),
                "np",
                None,
            )
            is_all_spec_decode = bool(
                draft_lens is not None
                and num_reqs > 0
                and (draft_lens[:num_reqs] >= 0).all()
            )
            # Injected FULL keys are valid only for a pure speculative decode
            # verifier. Prompt prefill, PD mixed and ordinary decode preserve
            # the exact original eager route.
            if not is_all_decode or not is_all_spec_decode:
                force_eager = True

        result = _orig_determine(
            self,
            num_tokens,
            num_reqs,
            num_scheduled_tokens_np,
            max_num_scheduled_tokens,
            use_cascade_attn,
            allow_microbatching,
            force_eager,
            force_uniform_decode,
            force_has_lora,
            force_num_active_loras,
            num_encoder_reqs,
        )
        runtime_mode, descriptor = result[:2]
        self._dcut_nonuniform_full_batch = bool(
            full_multishape
            and runtime_mode == CUDAGraphMode.FULL
            and not descriptor.uniform
        )
        self._dcut_ragged_full_request_capacity = (
            int(descriptor.num_reqs)
            if self._dcut_nonuniform_full_batch
            and descriptor.num_reqs is not None
            else None
        )
        if full_multishape:
            _dcut_set_full_gdn_metadata_route(
                self,
                runtime_mode == CUDAGraphMode.FULL and descriptor.uniform,
            )
        return result

    def _pad_query_start_loc_for_fia(
        self,
        query_start_loc,
        num_tokens_padded,
        num_reqs_padded,
        num_reqs,
        cudagraph_runtime_mode=None,
        batch_desc_num_reqs=None,
    ):
        use_ragged_target_fia = bool(
            getattr(self, "_dcut_nonuniform_full_batch", False)
        )
        use_fixed_drafter_fia = bool(
            getattr(
                self,
                "_dcut_drafter_from_nonuniform_decode",
                False,
            )
        )
        if not use_ragged_target_fia and not use_fixed_drafter_fia:
            return _orig_pad_query_start_loc_for_fia(
                self,
                query_start_loc,
                num_tokens_padded,
                num_reqs_padded,
                num_reqs,
                cudagraph_runtime_mode,
                batch_desc_num_reqs,
            )

        from vllm_ascend.worker.utils import copy_snapshot_to_gpu

        request_capacity = (
            int(batch_desc_num_reqs)
            if batch_desc_num_reqs is not None
            else int(num_reqs_padded)
        )
        actual_tokens = int(query_start_loc.np[num_reqs])
        fia_rows = _dcut_ragged_full_fia_rows(
            actual_tokens,
            num_tokens_padded,
            num_reqs,
            request_capacity,
        )
        if use_fixed_drafter_fia:
            # The draft remains on its stock uniform FULL graph. Preserve that
            # already-working request axis; only the ragged target FIA becomes
            # live-row dynamic.
            query_start_loc.np[
                num_reqs + 1 : request_capacity + 1
            ] = actual_tokens
            query_start_loc.np[request_capacity + 1] = num_tokens_padded
            self._dcut_fixed_drafter_fia_request_capacity = request_capacity
            copy_snapshot_to_gpu(query_start_loc)
            return request_capacity + 1

        # GDN keeps its own fixed-capacity metadata buffers. Target FIA instead
        # sees real request boundaries plus one dummy request only when the token
        # bucket has padding. The graph key remains one-dimensional in tokens.
        if fia_rows > num_reqs:
            query_start_loc.np[num_reqs + 1] = num_tokens_padded
        copy_snapshot_to_gpu(query_start_loc)
        return fia_rows

    def _build_attention_metadata(self, *args, **kwargs):
        result = _orig_build_attention_metadata(self, *args, **kwargs)
        use_ragged_target_fia = bool(
            getattr(self, "_dcut_nonuniform_full_batch", False)
        )
        use_fixed_drafter_fia = bool(
            getattr(
                self,
                "_dcut_drafter_from_nonuniform_decode",
                False,
            )
        )
        if not use_ragged_target_fia and not use_fixed_drafter_fia:
            return result

        attn_metadata = result[0]
        if not isinstance(attn_metadata, dict):
            return result
        if use_fixed_drafter_fia:
            request_capacity = getattr(
                self, "_dcut_fixed_drafter_fia_request_capacity", None
            )
            route = "drafter"
        else:
            request_capacity = getattr(
                self, "_dcut_ragged_full_request_capacity", None
            )
            route = "target"
        if request_capacity is None:
            return result
        num_reqs = kwargs.get("num_reqs")
        if num_reqs is None and len(args) > 1:
            num_reqs = args[1]
        if num_reqs is None:
            raise RuntimeError(
                "D-Cut ragged FULL attention metadata is missing num_reqs"
            )
        num_reqs = int(num_reqs)
        max_fia_rows = int(request_capacity) + 1
        persistent = getattr(
            self, "_dcut_ragged_full_fia_block_tables", None
        )
        if persistent is None:
            persistent = {}
            self._dcut_ragged_full_fia_block_tables = persistent

        normalized_metadata = set()
        seen = {}
        for prefix, metadata in attn_metadata.items():
            block_tables = getattr(metadata, "block_tables", None)
            if block_tables is None:
                continue
            actual_seq_lengths_q = getattr(
                metadata, "actual_seq_lengths_q", None
            )
            if actual_seq_lengths_q is None:
                continue
            fia_rows = len(actual_seq_lengths_q)
            if use_fixed_drafter_fia:
                if fia_rows != max_fia_rows:
                    raise RuntimeError(
                        "D-Cut FULL drafter FIA request axis changed: "
                        f"rows={fia_rows}, expected={max_fia_rows}"
                    )
            elif not num_reqs <= fia_rows <= num_reqs + 1:
                raise RuntimeError(
                    "D-Cut ragged FULL FIA request rows do not match the live "
                    f"batch: requests={num_reqs}, rows={fia_rows}"
                )
            if fia_rows > max_fia_rows:
                raise RuntimeError(
                    "D-Cut ragged FULL FIA request rows exceed graph storage: "
                    f"rows={fia_rows}, capacity={max_fia_rows}"
                )
            metadata_id = id(metadata)
            if metadata_id not in normalized_metadata:
                if use_fixed_drafter_fia:
                    _dcut_pad_fixed_drafter_fia_seq_lens(
                        metadata,
                        num_reqs,
                        int(request_capacity),
                    )
                else:
                    # Only the optional target token-padding request needs a
                    # synthetic KV length. Fixed GDN rows never enter FIA.
                    _dcut_pad_fia_dummy_seq_len(
                        metadata,
                        num_reqs,
                        fia_rows,
                    )
                normalized_metadata.add(metadata_id)
            stable = seen.get(metadata_id)
            if stable is None:
                key = (
                    route,
                    str(prefix),
                    max_fia_rows,
                    tuple(block_tables.shape[1:]),
                    block_tables.dtype,
                    block_tables.device,
                )
                entry = persistent.get(key)
                expected_shape = (
                    max_fia_rows,
                    *block_tables.shape[1:],
                )
                if entry is None:
                    storage = torch.zeros(
                        expected_shape,
                        dtype=block_tables.dtype,
                        device=block_tables.device,
                    )
                    entry = (storage, {})
                    persistent[key] = entry
                else:
                    storage = entry[0]
                if tuple(storage.shape) != expected_shape:
                    raise RuntimeError(
                        "D-Cut ragged FULL FIA block-table shape changed: "
                        f"expected={expected_shape}, "
                        f"actual={tuple(storage.shape)}"
                    )
                copy_rows = min(fia_rows, int(block_tables.shape[0]))
                storage[:copy_rows].copy_(
                    block_tables[:copy_rows], non_blocking=True
                )
                if copy_rows < fia_rows:
                    storage[copy_rows:fia_rows].zero_()
                views = entry[1]
                stable = views.get(fia_rows)
                if stable is None:
                    stable = storage[:fia_rows]
                    views[fia_rows] = stable
                seen[metadata_id] = stable
            metadata.block_tables = stable
        return result

    def _dummy_run(self, num_tokens, *args, **kwargs):
        cudagraph_runtime_mode = kwargs.get("cudagraph_runtime_mode")
        if cudagraph_runtime_mode is None and len(args) > 1:
            cudagraph_runtime_mode = args[1]
        is_profile = kwargs.get("is_profile", False)
        if len(args) > 4:
            is_profile = args[4]
        is_graph_capturing = kwargs.get("is_graph_capturing", False)
        if len(args) > 9:
            is_graph_capturing = args[9]
        uniform_decode = kwargs.get("uniform_decode", False)
        if len(args) > 3:
            uniform_decode = args[3]
        piecewise_capture_dummy = _dcut_piecewise_capture_dummy_enabled(
            self,
            cudagraph_runtime_mode,
            is_profile=bool(is_profile),
            is_graph_capturing=bool(is_graph_capturing),
        )
        ragged_full_capture_dummy = _dcut_is_ragged_full_capture(
            self,
            cudagraph_runtime_mode,
            bool(uniform_decode),
            bool(is_graph_capturing),
        )
        skip_drafter = ragged_full_capture_dummy
        if skip_drafter and not getattr(
            self,
            "_dcut_logged_ragged_full_drafter_skip",
            False,
        ):
            logger.warning(
                "D-Cut: skipping drafter dummy runs for ragged FULL "
                "target-only graph capture."
            )
            self._dcut_logged_ragged_full_drafter_skip = True
        if is_graph_capturing:
            from vllm.config import CUDAGraphMode

            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                logger.warning(
                    "D-Cut: PIECEWISE capture dummy token_bucket=%d "
                    "uniform_decode=%s enabled=%s real_warmup=%s.",
                    num_tokens,
                    bool(uniform_decode),
                    piecewise_capture_dummy,
                    getattr(self, "_dcut_in_real_warmup", False),
                )
        previous_piecewise = getattr(
            self,
            "_dcut_piecewise_capture_dummy",
            False,
        )
        previous_full = getattr(
            self,
            "_dcut_ragged_full_capture_dummy",
            False,
        )
        self._dcut_piecewise_capture_dummy = piecewise_capture_dummy
        self._dcut_ragged_full_capture_dummy = ragged_full_capture_dummy
        try:
            return _dcut_call_dummy_without_drafter(
                self,
                _orig_dummy_run,
                num_tokens,
                args,
                kwargs,
                skip_drafter,
            )
        finally:
            self._dcut_piecewise_capture_dummy = previous_piecewise
            self._dcut_ragged_full_capture_dummy = previous_full

    def _should_build_dummy_attn_metadata(
        self,
        force_attention=False,
        is_profile=False,
        cudagraph_runtime_mode=None,
    ):
        return (
            _orig_should_build_dummy(
                self,
                force_attention,
                is_profile,
                cudagraph_runtime_mode,
            )
            or getattr(
                self,
                "_dcut_piecewise_capture_dummy",
                False,
            )
        )

    def _model_forward(self, num_tokens_padded, *args, **kwargs):
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadata,
        )

        capture_dummy = getattr(
            self,
            "_dcut_piecewise_capture_dummy",
            False,
        )
        ragged_full_capture_dummy = getattr(
            self,
            "_dcut_ragged_full_capture_dummy",
            False,
        )
        graph_safe = False
        forward_context = get_forward_context()
        scheduler_has_prefill = getattr(
            self,
            "_dcut_gdn_scheduler_has_prefill",
            None,
        )
        runtime_mode = getattr(
            forward_context,
            "cudagraph_runtime_mode",
            None,
        )
        batch_descriptor = getattr(
            forward_context,
            "batch_descriptor",
            None,
        )
        stock_uniform_full = bool(
            _dcut_full_decode_multishape_enabled(self.vllm_config)
            and getattr(
                self.compilation_config.cudagraph_mode,
                "name",
                None,
            )
            == "FULL_DECODE_ONLY"
            and getattr(runtime_mode, "name", runtime_mode) == "FULL"
            and getattr(batch_descriptor, "uniform", False)
        )
        native_gdn_batch = stock_uniform_full or (
            _dcut_gdn_has_prefill(forward_context)
            if scheduler_has_prefill is None
            else bool(scheduler_has_prefill)
        )
        if forward_context is not None:
            # The context may be reused across forwards. Never let prepared
            # eager state outlive the metadata values it was derived from.
            forward_context._dcut_gdn_eager_spec_state = None
            forward_context._dcut_gdn_native_batch = native_gdn_batch
            forward_context._dcut_gdn_full_graph_safe = False
            forward_context._dcut_gdn_recurrent_piecewise_safe = False

        from vllm.config import CUDAGraphMode

        full_gdn_graph = (
            forward_context is not None
            and getattr(self, "_has_gdn", False)
            and (ragged_full_capture_dummy or not native_gdn_batch)
            and _dcut_full_decode_multishape_enabled(self.vllm_config)
            and self.compilation_config.cudagraph_mode
            == CUDAGraphMode.FULL_DECODE_ONLY
            and getattr(forward_context, "cudagraph_runtime_mode", None)
            == CUDAGraphMode.FULL
        )
        initial_spec_rows = _dcut_initial_handoff_spec_rows(self)
        if full_gdn_graph:
            if (
                getattr(self, "pcp_size", 1) != 1
                or getattr(self, "dcp_size", 1) != 1
            ):
                raise RuntimeError(
                    "D-Cut ragged FULL_DECODE_ONLY GDN requires PCP=1 "
                    "and DCP=1"
                )
            graph_request_capacity = getattr(
                batch_descriptor,
                "num_reqs",
                None,
            )
            expected_capacity = min(
                int(num_tokens_padded),
                int(self.vllm_config.scheduler_config.max_num_seqs),
            )
            if (
                graph_request_capacity is None
                or int(graph_request_capacity) != expected_capacity
            ):
                raise RuntimeError(
                    "D-Cut ragged FULL descriptor has an inconsistent fixed "
                    "request capacity: "
                    f"token_bucket={num_tokens_padded}, "
                    f"descriptor_capacity={graph_request_capacity}, "
                    f"expected_capacity={expected_capacity}"
                )
            try:
                if ragged_full_capture_dummy:
                    graph_safe = _dcut_prepare_gdn_graph_capture(
                        forward_context,
                        num_tokens_padded,
                        GDNAttentionMetadata,
                        expected_capacity,
                        self.uniform_decode_query_len,
                    )
                else:
                    graph_safe = _dcut_prepare_gdn_piecewise_replay(
                        forward_context,
                        num_tokens_padded,
                        GDNAttentionMetadata,
                        expected_capacity,
                        clear_unused_rows=True,
                        initial_spec_rows=initial_spec_rows,
                    )
            except Exception as exc:
                forward_context._dcut_gdn_full_graph_safe = False
                raise RuntimeError(
                    "D-Cut could not prepare fixed GDN metadata for FULL "
                    f"token bucket {num_tokens_padded}"
                ) from exc
            if not graph_safe:
                forward_context._dcut_gdn_full_graph_safe = False
                raise RuntimeError(
                    "D-Cut FULL_DECODE_ONLY GDN graph requires a pure "
                    "speculative decode batch"
                )
            forward_context._dcut_gdn_full_graph_safe = True

        if (
            self._dcut_gdn_piecewise_enabled
            and forward_context is not None
            and (capture_dummy or not native_gdn_batch)
            and getattr(
                forward_context, "cudagraph_runtime_mode", None
            )
            == CUDAGraphMode.PIECEWISE
        ):
            if (
                getattr(self, "pcp_size", 1) == 1
                and getattr(self, "dcp_size", 1) == 1
            ):
                try:
                    if capture_dummy:
                        graph_safe = (
                            _dcut_prepare_gdn_graph_capture(
                                forward_context,
                                num_tokens_padded,
                                GDNAttentionMetadata,
                                self.vllm_config.scheduler_config.max_num_seqs,
                                self.uniform_decode_query_len,
                            )
                        )
                    else:
                        graph_safe = _dcut_prepare_gdn_piecewise_replay(
                            forward_context,
                            num_tokens_padded,
                            GDNAttentionMetadata,
                            self.vllm_config.scheduler_config.max_num_seqs,
                            clear_unused_rows=True,
                            initial_spec_rows=initial_spec_rows,
                        )
                except Exception as exc:
                    logger.warning(
                        "D-Cut: graphable PIECEWISE GDN metadata "
                        "preparation failed; whole GDN remains eager: %s",
                        exc,
                    )
            elif not getattr(
                self, "_dcut_gdn_parallel_fallback_logged", False
            ):
                logger.warning(
                    "D-Cut: graphable PIECEWISE GDN is disabled "
                    "for PCP/DCP (pcp=%d, dcp=%d); whole GDN remains eager.",
                    getattr(self, "pcp_size", 1),
                    getattr(self, "dcp_size", 1),
                )
                self._dcut_gdn_parallel_fallback_logged = True
            forward_context._dcut_gdn_recurrent_piecewise_safe = graph_safe

        if (
            forward_context is not None
            and not graph_safe
            and not native_gdn_batch
        ):
            try:
                _dcut_prepare_gdn_eager_state(
                    forward_context,
                    GDNAttentionMetadata,
                    initial_spec_rows=initial_spec_rows,
                )
            except Exception as exc:
                forward_context._dcut_gdn_eager_spec_state = None
                if not getattr(
                    self, "_dcut_gdn_eager_prepare_fallback_logged", False
                ):
                    logger.warning(
                        "D-Cut: eager GDN shared-state preparation failed; "
                        "falling back to per-layer preparation: %s",
                        exc,
                    )
                    self._dcut_gdn_eager_prepare_fallback_logged = True

        if capture_dummy and not graph_safe:
            raise RuntimeError(
                "D-Cut could not build pure speculative GDN metadata "
                f"while capturing PIECEWISE token bucket "
                f"{num_tokens_padded}"
            )

        self._dcut_last_num_tokens_padded = num_tokens_padded
        self._dcut_last_graph_safe = graph_safe
        runtime_mode = (
            getattr(forward_context, "cudagraph_runtime_mode", None)
            if forward_context is not None
            else None
        )
        self._dcut_last_runtime_mode = (
            getattr(runtime_mode, "name", None)
            or (str(runtime_mode) if runtime_mode is not None else "UNKNOWN")
        )
        result = _orig_model_forward(
            self, num_tokens_padded, *args, **kwargs
        )
        if ragged_full_capture_dummy:
            logger.warning(
                "D-Cut: captured ragged FULL GDN graph through the "
                "expanded graphable recurrent path for token bucket %d "
                "(request_capacity=%s)",
                num_tokens_padded,
                getattr(batch_descriptor, "num_reqs", None),
            )
        if capture_dummy and self._dcut_gdn_piecewise_enabled:
            logger.warning(
                "D-Cut: captured PIECEWISE GDN graph with recurrent update "
                "inside for token bucket %d",
                num_tokens_padded,
            )
        return result

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        debug_stats = bool(os.environ.get(ENV_DEBUG_STATS))
        self._dcut_debug_stats_enabled = debug_stats
        if debug_stats:
            import time as _time

            _t_entry = _time.perf_counter()
            _last_end = getattr(self, "_dcut_last_debug_end", None)
            _has_gap_sample = _last_end is not None
            _gap_after_prev_ms = (
                (_t_entry - _last_end) * 1000
                if _has_gap_sample
                else 0.0
            )
            _prev_step = getattr(self, "_dcut_last_debug_step", None)
            _stats_io_after_prev_ms = getattr(
                self,
                "_dcut_last_stats_io_ms",
                0.0,
            )
            _rank_info = getattr(self, "_dcut_debug_rank_info", None)
            if _rank_info is None:
                _rank_info = _dcut_debug_rank_info()
                self._dcut_debug_rank_info = _rank_info
        # Drafter proposal is a separate worker call. Reset the per-step
        # marker so a skipped proposal cannot leak an old request set.
        self._dcut_zero_draft_handoffs_for_proposal = frozenset()
        if os.environ.get(ENV_FULL_DECODE_ONLY):
            return _orig_exec(self, scheduler_output, intermediate_tensors)

        _ctrl = getattr(self, "_verify_adaptive_controller", None)
        dcut_enabled = _ctrl is not None and not _env_flag(ENV_DISABLE)
        _recompute_handoff = _dcut_is_recompute_handoff(scheduler_output)
        _recompute_placeholder_req_ids = (
            _dcut_recompute_placeholder_req_ids(scheduler_output)
        )
        _adaptive_handoff_graph = _dcut_adaptive_handoff_graph_enabled(self)
        _adaptive_recompute_handoff = bool(
            dcut_enabled
            and _recompute_handoff
            and _recompute_placeholder_req_ids
            and _adaptive_handoff_graph
        )
        _native_recompute_handoff = bool(
            _recompute_handoff and not _adaptive_recompute_handoff
        )
        if debug_stats:
            _adaptive_probs_process_ms = 0.0
            _drafter_enable_ms = 0.0
            _truncate_ms = 0.0
            _prob_capture_bypass_ms = 0.0
            _prob_capture_reset_ms = 0.0
        if _native_recompute_handoff:
            # RecomputeScheduler's placeholder spec tokens have no drafter
            # probabilities. Bypass the complete D-Cut control/data path for
            # the whole handoff batch and use the native GDN implementation.
            self._dcut_capture_mixed_probs = False
            self._dcut_skip_current_prob_capture = True
            if debug_stats:
                _classify_ms = (
                    _time.perf_counter() - _t_entry
                ) * 1000
                _t_component = _time.perf_counter()
            _dcut_bypass_prob_capture_for_prefill(self)
            if debug_stats:
                _prob_capture_bypass_ms = (
                    _time.perf_counter() - _t_component
                ) * 1000
            if not getattr(self, "_dcut_logged_recompute_handoff", False):
                logger.info(
                    "D-Cut: PD recompute handoff detected; bypassing cut and "
                    "using native GDN metadata with stock graph dispatch."
                )
                self._dcut_logged_recompute_handoff = True
            # Preserve the zero-overhead production bypass. Debug mode keeps
            # going through the common timing/JSON path so the native handoff
            # forward is not incorrectly charged to the next step's gap.
            if not debug_stats:
                return _dcut_execute_native_recompute_handoff(
                    self,
                    _orig_exec,
                    scheduler_output,
                    intermediate_tensors,
                )
        elif _adaptive_recompute_handoff:
            self._dcut_capture_mixed_probs = False
            self._dcut_skip_current_prob_capture = False
            if not getattr(
                self,
                "_dcut_logged_adaptive_recompute_handoff",
                False,
            ):
                logger.info(
                    "D-Cut: routing RecomputeScheduler placeholders through "
                    "the adaptive PIECEWISE/FULL verifier with zero draft "
                    "caps."
                )
                self._dcut_logged_adaptive_recompute_handoff = True

        _zero_draft_handoffs = frozenset()
        if (
            dcut_enabled
            and not _recompute_handoff
            and _dcut_full_decode_multishape_enabled(self.vllm_config)
            and getattr(
                self.compilation_config.cudagraph_mode,
                "name",
                None,
            ) == "FULL_DECODE_ONLY"
        ):
            _zero_draft_handoffs = _dcut_zero_draft_kv_handoff_req_ids(
                self,
                scheduler_output,
            )
        _prefill_exempt_req_ids = _zero_draft_handoffs
        if _adaptive_recompute_handoff:
            _prefill_exempt_req_ids = (
                _prefill_exempt_req_ids
                | _recompute_placeholder_req_ids
            )
        self._dcut_zero_draft_handoffs_for_proposal = (
            _prefill_exempt_req_ids
        )
        _has_prefill = _dcut_has_prefill(
            self,
            scheduler_output,
            _prefill_exempt_req_ids,
        )
        _has_spec = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
        # Capture trim info before truncation only when optional debug timing is
        # enabled.  The regular D-Cut trim logger already records verify-token
        # reduction inside _dcut_truncate; keeping a second unconditional stats
        # path here adds Python work to every decode iteration.
        _full_draft = 0
        _batch_size = 0
        _spec_batch_size = 0
        _full_num_tokens = 0
        if debug_stats:
            _num_scheduled = getattr(
                scheduler_output,
                "num_scheduled_tokens",
                {},
            )
            _orig_spec = getattr(
                scheduler_output,
                "scheduled_spec_decode_tokens",
                {},
            )
            _full_draft = sum(len(t) for t in _orig_spec.values())
            _batch_size = len(_num_scheduled)
            _spec_batch_size = len(_orig_spec)
            _full_num_tokens = int(
                getattr(scheduler_output, "total_num_scheduled_tokens", 0)
            )
        if not _native_recompute_handoff:
            self._dcut_capture_mixed_probs = bool(
                _ctrl is not None and _has_prefill and _has_spec
            )
            self._dcut_skip_current_prob_capture = bool(
                _ctrl is not None and _has_prefill and not _has_spec
            )
        if debug_stats and not _native_recompute_handoff:
            _classify_ms = (_time.perf_counter() - _t_entry) * 1000
        if (
            _ctrl is not None
            and _has_prefill
            and not _native_recompute_handoff
        ):
            if debug_stats:
                _t_component = _time.perf_counter()
            _dcut_bypass_prob_capture_for_prefill(self)
            if debug_stats:
                _prob_capture_bypass_ms = (
                    _time.perf_counter() - _t_component
                ) * 1000

        if (
            _ctrl is not None
            and not _has_prefill
            and not _native_recompute_handoff
        ):
            if getattr(self, "_adaptive_probs_pending", False):
                if debug_stats:
                    _t_component = _time.perf_counter()
                try:
                    _maybe_process_adaptive_probs(
                        self,
                        stage="pre_truncate",
                    )
                except Exception as e:
                    logger.warning("D-Cut: process probs failed: %s", e)
                    self._adaptive_probs_pending = False
                    self._adaptive_probs_source = "process_error"
                    _ctrl.clear_adaptive_decision()
                finally:
                    if debug_stats:
                        _adaptive_probs_process_ms = (
                            _time.perf_counter() - _t_component
                        ) * 1000
            if debug_stats:
                _t_component = _time.perf_counter()
            _dcut_enable_drafter_probs(self)
            if debug_stats:
                _drafter_enable_ms = (
                    _time.perf_counter() - _t_component
                ) * 1000
            if dcut_enabled:
                if _adaptive_recompute_handoff:
                    _dcut_apply_zero_prob_recompute_caps(
                        _ctrl,
                        scheduler_output,
                        _recompute_placeholder_req_ids,
                    )
                if debug_stats:
                    _t_component = _time.perf_counter()
                scheduler_output = _dcut_truncate(
                    self,
                    scheduler_output,
                    has_prefill=_has_prefill,
                    zero_draft_handoff_req_ids=_zero_draft_handoffs,
                )
                if debug_stats:
                    _truncate_ms = (
                        _time.perf_counter() - _t_component
                    ) * 1000
            if debug_stats:
                _t_component = _time.perf_counter()
            if dcut_enabled and _zero_draft_handoffs:
                scheduler_output = _dcut_add_zero_draft_handoffs(
                    scheduler_output,
                    _zero_draft_handoffs,
                )
                _current_spec = getattr(
                    scheduler_output,
                    "scheduled_spec_decode_tokens",
                    None,
                )
                _has_spec = bool(_current_spec)
                if debug_stats:
                    _spec_batch_size = len(_current_spec or {})
                if not getattr(
                    self,
                    "_dcut_logged_zero_draft_kv_handoff",
                    False,
                ):
                    logger.info(
                        "D-Cut: routing KV-consumer first-token handoffs as "
                        "zero-draft rows in the ragged FULL graph."
                    )
                    self._dcut_logged_zero_draft_kv_handoff = True
            _dcut_prepare_prob_capture(self, scheduler_output)
            if debug_stats:
                _prob_capture_reset_ms = (
                    _time.perf_counter() - _t_component
                ) * 1000

        if not debug_stats:
            return _dcut_execute_with_gdn_prefill_route(
                self,
                _orig_exec,
                scheduler_output,
                intermediate_tensors,
                _has_prefill,
            )

        # Optional slow-path debug timing.  Keep it behind an env gate because
        # perf_counter plus per-step Python aggregation is visible at high ITL.
        _kept_draft = _full_draft
        if (
            dcut_enabled
            and _has_spec
            and not _native_recompute_handoff
        ):
            _new_spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
            _kept_draft = sum(len(t) for t in _new_spec.values())
        _num_tokens_actual = int(
            getattr(scheduler_output, "total_num_scheduled_tokens", 0)
        )
        import torch as _torch

        _t_pre_cpu_end = _time.perf_counter()
        _pre_cpu_total_ms = (_t_pre_cpu_end - _t_entry) * 1000
        _pre_cpu_other_ms = max(
            0.0,
            _pre_cpu_total_ms
            - _classify_ms
            - _adaptive_probs_process_ms
            - _drafter_enable_ms
            - _truncate_ms
            - _prob_capture_bypass_ms
            - _prob_capture_reset_ms,
        )
        _torch.npu.synchronize()
        _t_execute_start = _time.perf_counter()
        _pre_sync_ms = (_t_execute_start - _t_pre_cpu_end) * 1000
        if _native_recompute_handoff:
            result = _dcut_execute_native_recompute_handoff(
                self,
                _orig_exec,
                scheduler_output,
                intermediate_tensors,
            )
        else:
            result = _dcut_execute_with_gdn_prefill_route(
                self,
                _orig_exec,
                scheduler_output,
                intermediate_tensors,
                _has_prefill,
            )
        _t_execute_return = _time.perf_counter()
        _execute_call_ms = (
            _t_execute_return - _t_execute_start
        ) * 1000
        _torch.npu.synchronize()
        _t_fwd_end = _time.perf_counter()
        _post_sync_ms = (_t_fwd_end - _t_execute_return) * 1000
        _fwd_ms = (_t_fwd_end - _t_execute_start) * 1000
        if not hasattr(self, "_dcut_fwd_accum"):
            self._dcut_fwd_accum = {
                "full": 0,
                "kept": 0,
                "cut": 0,
                "fwd_ms": 0.0,
                "gap_after_prev_ms": 0.0,
                "gap_samples": 0,
                "steps": 0,
                "spec_steps": 0,
            }
        acc = self._dcut_fwd_accum
        acc["steps"] += 1
        acc["fwd_ms"] += _fwd_ms
        if _has_gap_sample:
            acc["gap_after_prev_ms"] += _gap_after_prev_ms
            acc["gap_samples"] += 1
        if _has_spec and not _native_recompute_handoff:
            acc["spec_steps"] += 1
            acc["full"] += _full_draft
            acc["kept"] += _kept_draft
            acc["cut"] += (_full_draft - _kept_draft)
        if acc["steps"] % 50 == 0 and _rank_info["is_writer"]:
            _avg_fwd = acc["fwd_ms"] / acc["steps"]
            _avg_gap = (
                acc["gap_after_prev_ms"] / acc["gap_samples"]
                if acc["gap_samples"]
                else 0.0
            )
            if acc["full"] > 0:
                _cut_pct = 100.0 * acc["cut"] / acc["full"]
                logger.warning(
                    "D-Cut step %d: full_draft=%d cut=%d (%.1f%%) "
                    "kept=%d avg_fwd=%.1fms avg_inter_call_gap=%.1fms",
                    acc["steps"],
                    acc["full"],
                    acc["cut"],
                    _cut_pct,
                    acc["kept"],
                    _avg_fwd,
                    _avg_gap,
                )
            else:
                logger.warning(
                    "D-Cut step %d: no spec reqs, avg_fwd=%.1fms "
                    "avg_inter_call_gap=%.1fms",
                    acc["steps"],
                    _avg_fwd,
                    _avg_gap,
                )

        _t_post_cpu_end = _time.perf_counter()
        _post_cpu_ms = (_t_post_cpu_end - _t_fwd_end) * 1000
        _fwd_stats_out = os.environ.get("VLLM_DCUT_FWD_STATS_OUT")
        _stats_io_ms = 0.0
        if _fwd_stats_out and _rank_info["is_writer"]:
            import json as _json

            _num_padded = getattr(self, "_dcut_last_num_tokens_padded", 0)
            _graph_safe = getattr(self, "_dcut_last_graph_safe", False)
            _runtime_mode = getattr(
                self,
                "_dcut_last_runtime_mode",
                "UNKNOWN",
            )
            _is_eager = _runtime_mode in {
                "NONE",
                "EAGER",
            }
            _t_stats_io_start = _time.perf_counter()
            _entry = {
                "step": acc["steps"],
                "pid": os.getpid(),
                "world_rank": _rank_info["world_rank"],
                "tp_rank": _rank_info["tp_rank"],
                "pp_rank": _rank_info["pp_rank"],
                "bs": _batch_size,
                "spec_bs": _spec_batch_size,
                "has_prefill": _has_prefill,
                "has_spec": _has_spec,
                "mixed_batch": _has_prefill and _has_spec,
                "pure_prefill": _has_prefill and not _has_spec,
                "decode_only": not _has_prefill,
                "dcut_enabled": dcut_enabled,
                "recompute_handoff": _recompute_handoff,
                "recompute_placeholder_dcut": (
                    _adaptive_recompute_handoff
                ),
                "recompute_placeholder_count": len(
                    _recompute_placeholder_req_ids
                ),
                "zero_draft_handoff_count": len(
                    _zero_draft_handoffs
                ),
                "reused_handoff_decision": bool(
                    getattr(
                        self,
                        "_dcut_last_reused_handoff_decision",
                        False,
                    )
                ),
                "reused_survivor_decision": bool(
                    getattr(
                        self,
                        "_dcut_last_reused_survivor_decision",
                        False,
                    )
                ),
                "dcut_bypassed": bool(
                    _native_recompute_handoff or _has_prefill
                ),
                "dcut_bypass_reason": (
                    "recompute_handoff"
                    if _native_recompute_handoff
                    else "recompute_placeholder_dcut"
                    if _adaptive_recompute_handoff
                    else "prefill"
                    if _has_prefill
                    else "none"
                ),
                "prob_capture_enabled": (
                    _ctrl is not None
                    and not self._dcut_skip_current_prob_capture
                ),
                "mixed_prob_capture_planned": bool(
                    getattr(self, "_dcut_capture_mixed_probs", False)
                ),
                "drafter_needs_draft_probs": bool(
                    getattr(
                        getattr(self, "drafter", None),
                        "needs_draft_probs",
                        False,
                    )
                ),
                "draft_ran_python": bool(
                    getattr(
                        getattr(self, "drafter", None),
                        "_dcut_last_draft_ran_python",
                        False,
                    )
                ),
                "adaptive_probs_pending_after_step": bool(
                    getattr(self, "_adaptive_probs_pending", False)
                ),
                "prob_decision_source": getattr(
                    self,
                    "_adaptive_probs_last_consumed_source",
                    "none",
                ),
                "prob_decision_generation": int(
                    getattr(self, "_adaptive_probs_last_consumed_generation", 0)
                ),
                "prob_pending_source": getattr(
                    self,
                    "_adaptive_probs_source",
                    "none",
                ),
                "prob_pending_generation": int(
                    getattr(self, "_adaptive_probs_generation", 0)
                ),
                "prob_decision_mean_by_position": getattr(
                    self,
                    "_adaptive_probs_last_consumed_mean_by_position",
                    [],
                ),
                "prob_capture_skipped_for_prefill": (
                    self._dcut_skip_current_prob_capture
                ),
                "full_draft": _full_draft,
                "kept_draft": _kept_draft,
                "trimmed": _full_draft - _kept_draft,
                "cut_applied": _full_draft != _kept_draft,
                "full_num_tokens": _full_num_tokens,
                "num_tokens_actual": _num_tokens_actual,
                "num_tokens_padded": _num_padded,
                "runtime_mode": _runtime_mode,
                "is_eager": _is_eager,
                "gdn_native_path": bool(
                    _has_prefill or _native_recompute_handoff
                ),
                "gdn_graph_safe": _graph_safe,
                "prev_step": _prev_step,
                "gap_sample_valid": _has_gap_sample,
                "inter_call_gap_after_prev_step_ms": round(
                    _gap_after_prev_ms,
                    3,
                ),
                "stats_io_after_prev_step_ms": round(
                    _stats_io_after_prev_ms,
                    3,
                ),
                "classify_ms": round(_classify_ms, 3),
                "adaptive_probs_process_ms": round(
                    _adaptive_probs_process_ms,
                    3,
                ),
                "drafter_enable_ms": round(_drafter_enable_ms, 3),
                "truncate_ms": round(_truncate_ms, 3),
                "prob_capture_bypass_ms": round(
                    _prob_capture_bypass_ms,
                    3,
                ),
                "prob_capture_reset_ms": round(
                    _prob_capture_reset_ms,
                    3,
                ),
                "pre_cpu_other_ms": round(_pre_cpu_other_ms, 3),
                "pre_cpu_total_ms": round(_pre_cpu_total_ms, 3),
                "pre_sync_ms": round(_pre_sync_ms, 3),
                "pre_total_ms": round(
                    _pre_cpu_total_ms + _pre_sync_ms,
                    3,
                ),
                "execute_call_ms": round(_execute_call_ms, 3),
                "post_sync_ms": round(_post_sync_ms, 3),
                "fwd_ms": round(_fwd_ms, 2),
                "post_cpu_ms": round(_post_cpu_ms, 3),
            }
            try:
                with open(_fwd_stats_out, "a") as _f:
                    _f.write(_json.dumps(_entry) + chr(10))
            except Exception:
                pass
            _stats_io_ms = (
                _time.perf_counter() - _t_stats_io_start
            ) * 1000
        self._dcut_last_stats_io_ms = _stats_io_ms
        self._dcut_last_debug_step = acc["steps"]
        self._dcut_last_debug_end = _time.perf_counter()
        return result

    _orig_sample_tokens = R.sample_tokens

    def sample_tokens(self, *a, **k):
        # The target side of a mixed prefill/decode batch stays on the native
        # eager path. Re-enable selected-prob collection only for the draft
        # proposal produced afterwards: those exact tokens are verified by the
        # next decode-only iteration, which otherwise has to run uncut.
        if getattr(self, "_dcut_capture_mixed_probs", False):
            _dcut_enable_drafter_probs(self)
        out = _orig_sample_tokens(self, *a, **k)
        if (
            os.environ.get(ENV_FULL_DECODE_ONLY)
            or getattr(self, "_dcut_skip_current_prob_capture", False)
        ):
            return out
        if (
            getattr(self, "_adaptive_probs_pending", False)
            and not getattr(self, "_dcut_skip_unready_probs", False)
            and getattr(
                self,
                "_dcut_process_probs_stage",
                "pre_truncate",
            )
            == "post_sample"
        ):
            try:
                _maybe_process_adaptive_probs(self, stage="post_sample")
            except Exception as e:
                logger.warning("D-Cut: process probs failed: %s", e)
                self._adaptive_probs_pending = False
                self._adaptive_probs_source = "process_error"
                controller = getattr(self, "_verify_adaptive_controller", None)
                if controller is not None:
                    controller.clear_adaptive_decision()
        return out

    _orig_copy = R._copy_draft_token_ids_to_cpu

    def _copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
        _orig_copy(self, scheduler_output, zeros_only)
        if (
            os.environ.get(ENV_FULL_DECODE_ONLY)
            or getattr(self, "_dcut_skip_current_prob_capture", False)
        ):
            return
        if getattr(self, "_verify_adaptive_controller", None) is not None:
            try:
                _dcut_queue_probs(self, zeros_only)
            except Exception as e:
                logger.warning("D-Cut: queue probs failed: %s", e)
                self._adaptive_probs_pending = False
                self._adaptive_probs_source = "queue_error"
                controller = getattr(self, "_verify_adaptive_controller", None)
                if controller is not None:
                    controller.clear_adaptive_decision()

    _orig_update = R._update_states

    def _update_states(self, scheduler_output):
        ret = _orig_update(self, scheduler_output)
        ctrl = getattr(self, "_verify_adaptive_controller", None)
        if ctrl is not None:
            for rid in scheduler_output.finished_req_ids:
                ctrl.invalidate(rid)
        return ret

    R.__init__ = __init__
    R._model_forward = _model_forward
    R._dummy_run = _dummy_run
    R._should_build_dummy_attn_metadata = _should_build_dummy_attn_metadata
    R._determine_batch_execution_and_padding = (
        _determine_batch_execution_and_padding
    )
    R._pad_query_start_loc_for_fia = _pad_query_start_loc_for_fia
    R._build_attention_metadata = _build_attention_metadata
    R.execute_model = execute_model
    R.sample_tokens = sample_tokens
    R._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    R._update_states = _update_states
    R._adaptive_profile_run = _adaptive_profile_run
    R._adaptive_profile_draft_run = _adaptive_profile_draft_run
    R.profile_adaptive_cost = profile_adaptive_cost
    R._maybe_process_adaptive_probs = _maybe_process_adaptive_probs
    R._dcut_enable_drafter_probs = _dcut_enable_drafter_probs
    R._dcut_patched = True

    logger.info(
        "D-Cut: using graph-captured GDN in the vLLM 0.23 PIECEWISE path."
    )
