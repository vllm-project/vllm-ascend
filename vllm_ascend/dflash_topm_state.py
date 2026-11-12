"""Graph-safe activation-routing state for DFlash and DSpark verification."""

import json
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from vllm.distributed import get_tp_group
from vllm.logger import logger


_activation_batch: ContextVar[tuple | None] = ContextVar("activation_batch", default=None)
_active_forward: ContextVar[object | None] = ContextVar("active_forward", default=None)


def set_activation_routing_batch(state, verify_rows, segments):
    """Publish scheduler-derived batch metadata to MRv2's forward context."""
    _activation_batch.set((state, tuple(verify_rows), list(segments)) if state is not None else None)


def get_activation_routing_batch():
    return _activation_batch.get()


def finish_activation_routing():
    context = _active_forward.get()
    if context is not None:
        state = getattr(context, "dflash_topm_state", None)
        if state is not None:
            state.finish(context)
    _active_forward.set(None)


def _context_value(context, name, default=None):
    value = getattr(context, name, None)
    if value is not None:
        return value
    return getattr(context, "additional_kwargs", {}).get(name, default)


class DFlashTopMState:
    def __init__(self, config_path):
        self._init_from_config(json.loads(Path(config_path).read_text()))

    @classmethod
    def from_dict(cls, config, text_config=None):
        """Build the state from an additional_config-style dict.

        num_experts / route_top_k / scoring_func are deliberately not part of
        the config section: when ``text_config`` (the model's hf text config)
        is given they are derived from it, removing a second source of truth.
        """
        state = cls.__new__(cls)
        config = dict(config)
        if text_config is not None:
            config["num_experts"] = int(
                getattr(
                    text_config,
                    "n_routed_experts",
                    getattr(text_config, "num_experts", 128),
                )
            )
            config["route_top_k"] = int(getattr(text_config, "num_experts_per_tok", 8))
            config["scoring_func"] = getattr(text_config, "scoring_func", "softmax")
        state._init_from_config(config)
        return state

    def _init_from_config(self, config):
        self.config = config
        self.backend = self.config.get("backend", "baseline")
        if self.backend == "anchor_union_topm":
            raise ValueError(
                "Top-M remains an unvalidated numerical prototype; use "
                "anchor_union_fused_native for this verified candidate. "
                "Prototype kernels remain in the experiment's kernels_dev directory."
            )
        if self.backend not in {
            "baseline",
            "anchor_union_reference",
            "anchor_union_fused_native",
        }:
            raise ValueError(f"Unknown Top-M backend: {self.backend}")
        self.verify_block_size = int(self.config.get("verify_block_size", 16))
        self.num_experts = int(self.config.get("num_experts", 128))
        self.route_top_k = int(self.config.get("route_top_k", 8))
        self.scoring_func = self.config.get("scoring_func", "softmax")
        self.protected_rows = int(self.config.get("protected_rows", 3))
        self.suffix_pool_top_k = int(self.config.get("suffix_pool_top_k", 2))
        expected_router_layers = self.config.get("expected_router_layers", 48)
        self.expected_router_layers = None if expected_router_layers is None else int(expected_router_layers)
        self.fused_rows = tuple(int(value) for value in self.config.get("fused_rows", [16, 32, 48, 64]))
        if self.verify_block_size < 2:
            raise ValueError("verify_block_size must be at least 2")
        if self.num_experts not in (128, 256):
            raise ValueError("num_experts must be 128 or 256")
        if not 1 <= self.route_top_k <= 8:
            raise ValueError("route_top_k must be in [1, 8]")
        if self.scoring_func not in {"softmax", "sigmoid", "sqrtsoftplus"}:
            raise ValueError("scoring_func must be softmax, sigmoid, or sqrtsoftplus")
        if not 1 <= self.protected_rows <= self.verify_block_size:
            raise ValueError("protected_rows must be in [1, verify_block_size]")
        if not 1 <= self.suffix_pool_top_k <= self.route_top_k:
            raise ValueError("suffix_pool_top_k must be in [1, route_top_k]")
        if (
            self.backend == "anchor_union_fused_native"
            and self.scoring_func == "sqrtsoftplus"
            and self.suffix_pool_top_k != 2
        ):
            raise ValueError(
                "anchor_union_fused_native requires suffix_pool_top_k=2 for "
                "the DSpark contract; use anchor_union_reference to "
                "experiment with other pool sizes."
            )
        if any(value <= 0 for value in self.fused_rows):
            raise ValueError("fused_rows must contain positive token counts")
        self.trace_dir = self.config.get("trace_dir")
        self.trace_limit = int(self.config.get("trace_steps_per_bs", 32))
        self.counts = {}
        self.sequence = 0
        self.roles_gpu = {}
        self.roles_cpu = {}
        self.workspaces = {}
        self.current_roles = None
        self.has_verify = False
        self.reference_masks = {}
        self.profile_sequence = 0
        self.prepared_num_tokens = 0
        self.logged_route_contracts = set()

    def should_log_route_contract(self, contract):
        if contract in self.logged_route_contracts:
            return False
        self.logged_route_contracts.add(contract)
        return True

    def prepare(self, num_tokens, segments, device):
        """Host scheduler metadata -> persistent device role buffer, before replay.

        This never reads a device tensor on the host. Padding and prefill have
        role zero and therefore cannot enlarge the candidate pool.
        """
        self.has_verify = bool(segments)
        self.prepared_num_tokens = num_tokens
        self.current_roles = None
        if self.backend == "baseline" or not segments:
            return
        if num_tokens not in self.roles_gpu:
            self.roles_cpu[num_tokens] = torch.zeros(num_tokens, dtype=torch.int32, pin_memory=True)
            self.roles_gpu[num_tokens] = torch.empty(num_tokens, dtype=torch.int32, device=device)
        cpu = self.roles_cpu[num_tokens]
        cpu.zero_()
        for start, length in segments:
            if start < 0 or length < 1 or start + length > num_tokens:
                raise ValueError("Activation-routing segment outside padded token buffer")
            protected = min(self.protected_rows, length)
            cpu[start : start + protected] = 1
            cpu[start + protected : start + length] = 2
        gpu = self.roles_gpu[num_tokens]
        gpu.copy_(cpu, non_blocking=True)
        self.current_roles = gpu

    def _routing_scores(self, logits):
        if self.scoring_func == "softmax":
            return logits.softmax(dim=-1)
        if self.scoring_func == "sigmoid":
            return logits.sigmoid()
        if self.scoring_func == "sqrtsoftplus":
            return F.softplus(logits).sqrt()
        raise AssertionError(f"Unexpected scoring function: {self.scoring_func}")

    def route_reference(
        self,
        logits,
        roles,
        top_k,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
    ):
        """Graph-safe exact anchor-union fallback for the model's route rule."""
        if logits.ndim != 2 or logits.shape[0] != roles.numel():
            raise ValueError("Router rows must match the current scheduler role buffer")
        if logits.shape[1] != self.num_experts:
            raise ValueError(
                f"Router expert count does not match activation-routing config: {logits.shape[1]} != {self.num_experts}"
            )
        if top_k != self.route_top_k:
            raise ValueError(f"Router top_k does not match activation-routing config: {top_k} != {self.route_top_k}")

        routing_scores = self._routing_scores(logits)
        selection_scores = routing_scores
        if e_score_correction_bias is not None:
            selection_scores = routing_scores + e_score_correction_bias.unsqueeze(0)

        if logits.device.type == "cpu":
            initial_ids = torch.topk(selection_scores, self.route_top_k, dim=-1).indices
            limits = torch.where(
                roles == 1,
                self.route_top_k,
                torch.where(roles == 2, self.suffix_pool_top_k, 0),
            )
            ranks = torch.arange(self.route_top_k, device=logits.device).unsqueeze(0)
            active = ranks < limits.unsqueeze(1)
            flags = torch.zeros_like(selection_scores, dtype=torch.bool)
            flags.scatter_(1, initial_ids, active)
            pool = flags.any(dim=0)
            selection_scores = selection_scores.masked_fill(
                (roles.unsqueeze(1) == 2) & ~pool.unsqueeze(0),
                torch.finfo(selection_scores.dtype).min,
            )
        else:
            from vllm_ascend.ops.triton.dflash_anchor_reference import (
                apply_anchor_reference,
            )

            key = (logits.device, self.num_experts)
            if key not in self.reference_masks:
                self.reference_masks[key] = torch.zeros(
                    self.num_experts,
                    device=logits.device,
                    dtype=torch.bool,
                )
            selection_scores = apply_anchor_reference(
                selection_scores,
                roles,
                self.reference_masks[key],
                self.route_top_k,
                self.suffix_pool_top_k,
            )
        _, topk_ids = torch.topk(selection_scores, top_k, dim=-1)
        topk_weights = routing_scores.gather(1, topk_ids)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * routed_scaling_factor
        return topk_weights, topk_ids.to(torch.int32)

    def supports_fused_route(
        self,
        logits,
        top_k,
        scoring_func,
        routed_scaling_factor,
        e_score_correction_bias,
    ):
        common = (
            self.backend == "anchor_union_fused_native"
            and logits.ndim == 2
            and logits.shape[0] in self.fused_rows
            and logits.shape[1] == self.num_experts
            and top_k == self.route_top_k
            and scoring_func == self.scoring_func
        )
        dspark_contract = (
            logits.dtype == torch.float32
            and self.num_experts == 256
            and self.route_top_k == 6
            and self.suffix_pool_top_k == 2
            and self.scoring_func == "sqrtsoftplus"
            and routed_scaling_factor == 1.5
            and e_score_correction_bias is not None
            and e_score_correction_bias.shape == (256,)
        )
        return common and dspark_contract

    def route(
        self,
        logits,
        roles,
        top_k,
        renormalize,
        scoring_func,
        routed_scaling_factor,
        e_score_correction_bias,
    ):
        if self.backend != "anchor_union_fused_native":
            raise ValueError(f"Backend {self.backend!r} does not provide a fused route")
        if scoring_func == "sqrtsoftplus":
            from vllm_ascend.ops.triton.dflash_fused_mask_native import (
                allocate_dspark_workspace,
                dspark_route,
            )

            key = ("dspark", logits.shape[0], logits.device, logits.dtype)
            if key not in self.workspaces:
                self.workspaces[key] = allocate_dspark_workspace(logits)
            return dspark_route(
                logits,
                roles,
                e_score_correction_bias,
                workspace=self.workspaces[key],
                route_top_k=top_k,
                suffix_pool_top_k=self.suffix_pool_top_k,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
            )

        raise ValueError(
            "The fused route supports the DeepSeek DSpark contract "
            "(sqrtsoftplus scoring) only; the reference implementation "
            "covers the remaining contracts."
        )

    def begin(self, context):
        layout = getattr(context, "dflash_graph_layout", None)
        if layout is not None and self.backend != "baseline":
            # Dummy graph warmup/capture executes this outside model.forward.
            num_tokens = int(getattr(context, "num_tokens", 0))
            segments = [
                (start, self.verify_block_size)
                for start in range(
                    0,
                    min(num_tokens, self.verify_block_size * (len(layout.req_boundaries) - 1)),
                    self.verify_block_size,
                )
            ]
            self.prepare(num_tokens, segments, layout.all_rows.device)
        verify_rows = getattr(context, "dflash_verify_rows", ())
        context.dflash_topm_roles = (
            self.current_roles if (layout is not None or verify_rows or self.has_verify) else None
        )
        context.dflash_topm_layers = None
        rows = verify_rows
        context.dflash_topm_profile_range = None
        if self.config.get("msprof") and rows:
            # Profiling only: stream-ordered markers identify real verify
            # forwards and their batch/layout without task-ID assumptions.
            lengths = ",".join(str(r.numel() + 1) for _, r in rows)
            graph_mode = getattr(context, "cudagraph_runtime_mode", "unknown")
            graph_mode = getattr(graph_mode, "name", str(graph_mode))
            label = (
                f"DFLASH_VERIFY|seq={self.profile_sequence}|bs={len(rows)}"
                f"|padded_tokens={self.prepared_num_tokens}|lengths={lengths}|mode={graph_mode}"
            )
            context.dflash_topm_profile_range = torch.npu.mstx.range_start(label, torch.npu.current_stream())
            self.profile_sequence += 1
        if not self.trace_dir or not rows or get_tp_group().rank_in_group != 0:
            return
        bs = len(rows)
        if self.counts.get(bs, 0) >= self.trace_limit:
            return
        # Trace mode only: intentional CPU sync to preserve exact real boundaries.
        segments = []
        for req_id, draft_rows in rows:
            draft = draft_rows.cpu().tolist()
            if len(draft) != self.verify_block_size - 1:
                return
            segments.append({"request_id": req_id, "rows": [draft[0] - 1] + draft})
        context.dflash_topm_segments = segments
        context.dflash_topm_layers = []
        context.dflash_topm_selected = []

    def record(self, context, logits):
        layers = getattr(context, "dflash_topm_layers", None)
        if layers is None:
            return
        rows = [i for seg in context.dflash_topm_segments for i in seg["rows"]]
        indices = torch.tensor(rows, device=logits.device, dtype=torch.long)
        layers.append(logits.index_select(0, indices).detach().clone())

    def record_selection(self, context, weights, ids):
        if getattr(context, "dflash_topm_layers", None) is None:
            return
        rows = [i for seg in context.dflash_topm_segments for i in seg["rows"]]
        indices = torch.tensor(rows, device=ids.device, dtype=torch.long)
        context.dflash_topm_selected.append(
            (
                weights.index_select(0, indices).detach().clone(),
                ids.index_select(0, indices).detach().clone(),
            )
        )

    def finish(self, context):
        profile_range = getattr(context, "dflash_topm_profile_range", None)
        if profile_range is not None:
            torch.npu.mstx.range_end(profile_range)
        layers = getattr(context, "dflash_topm_layers", None)
        if not layers:
            return
        if self.expected_router_layers is not None and len(layers) != self.expected_router_layers:
            raise RuntimeError(f"Expected {self.expected_router_layers} target router layers, got {len(layers)}")
        bs = len(context.dflash_topm_segments)
        path = Path(self.trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        selected = context.dflash_topm_selected
        if len(selected) != len(layers):
            raise RuntimeError("Trace must contain all executed routing outputs")
        torch.save(
            {
                "logits": torch.stack(layers).cpu(),
                "selected_weights": torch.stack([w for w, _ in selected]).cpu(),
                "selected_ids": torch.stack([ids for _, ids in selected]).cpu(),
                "segments": context.dflash_topm_segments,
                "verify_bs": bs,
                "sequence": self.sequence,
                "backend": self.backend,
            },
            path / f"step_{self.sequence:05d}_bs{bs}.pt",
        )
        self.sequence += 1
        self.counts[bs] = self.counts.get(bs, 0) + 1
        context.dflash_topm_layers = None


def try_activation_route(
    context,
    router_logits,
    hidden_states,
    top_k,
    renormalize,
    scoring_func,
    e_score_correction_bias,
    routed_scaling_factor,
    custom_routing_function,
    tid2eid,
    mix_placement,
    topk_group,
    num_expert_group,
):
    """Eligibility gate and dispatch for the select_experts hook.

    Returns the routed ``(topk_weights, topk_ids)`` pair when the current
    forward matches the activation-routing contract, or ``None`` to keep the
    default MoE routing path unchanged (fully off when no state is attached).
    """
    topm_state = _context_value(context, "dflash_topm_state")
    roles = _context_value(context, "dflash_topm_roles")
    if topm_state is not None:
        context.dflash_topm_state = topm_state
        context.dflash_topm_roles = roles
        context.dflash_verify_rows = _context_value(context, "dflash_verify_rows", ())
        if _active_forward.get() is not context:
            topm_state.begin(context)
            _active_forward.set(context)
    activation_routing_supported = (
        topm_state is not None
        and topm_state.backend != "baseline"
        and roles is not None
        and not _context_value(context, "is_draft_model", False)
        and router_logits.ndim == 2
        and router_logits.shape[0] == roles.numel()
        and router_logits.shape[1] == topm_state.num_experts
        and top_k == topm_state.route_top_k
        and renormalize
        and scoring_func == topm_state.scoring_func
        and custom_routing_function is None
        and tid2eid is None
        and not mix_placement
        and topk_group in (None, 1)
        and num_expert_group in (None, 1)
    )
    if topm_state is not None and roles is not None:
        route_contract = (
            tuple(router_logits.shape),
            str(router_logits.dtype),
            str(hidden_states.dtype),
            top_k,
            scoring_func,
            e_score_correction_bias is not None,
            routed_scaling_factor,
            tid2eid is not None,
            bool(_context_value(context, "is_draft_model", False)),
            activation_routing_supported,
        )
        if topm_state.should_log_route_contract(route_contract):
            logger.info(
                "[ACTIVATION-ROUTING] shape=%s router_dtype=%s hidden_dtype=%s top_k=%d "
                "scoring_func=%s bias=%s scale=%s tid2eid=%s draft=%s "
                "eligible=%s fused=%s",
                tuple(router_logits.shape),
                router_logits.dtype,
                hidden_states.dtype,
                top_k,
                scoring_func,
                e_score_correction_bias is not None,
                routed_scaling_factor,
                tid2eid is not None,
                bool(_context_value(context, "is_draft_model", False)),
                activation_routing_supported,
                activation_routing_supported
                and topm_state.supports_fused_route(
                    router_logits,
                    top_k,
                    scoring_func,
                    routed_scaling_factor,
                    e_score_correction_bias,
                ),
            )
    if not activation_routing_supported:
        return None
    topm_state.record(context, router_logits)
    if topm_state.supports_fused_route(
        router_logits,
        top_k,
        scoring_func,
        routed_scaling_factor,
        e_score_correction_bias,
    ):
        return topm_state.route(
            router_logits,
            roles,
            top_k,
            renormalize,
            scoring_func,
            routed_scaling_factor,
            e_score_correction_bias,
        )
    return topm_state.route_reference(
        router_logits,
        roles,
        top_k,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
    )


def build_verify_segments(scheduler_output, input_batch, num_scheduled_tokens_np, spec_decode_metadata):
    """Locate per-request verify segments inside the padded token buffer.

    Returns ``(verify_rows, segments)``: verify_rows pairs each request id with
    its draft-token row indices (trace/profiling only); segments are the
    ``(start, length)`` spans consumed by :meth:`DFlashTopMState.prepare`.
    """
    verify_rows = []
    segments = []
    if spec_decode_metadata is None:
        return tuple(verify_rows), segments
    cumulative_tokens = np.cumsum(num_scheduled_tokens_np, dtype=np.int32)
    for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
        req_idx = input_batch.req_id_to_index[req_id]
        draft_count = len(draft_token_ids)
        is_decode = (
            input_batch.num_computed_tokens_cpu[req_idx]
            >= input_batch.num_prompt_tokens[req_idx]
        )
        if not (draft_count and is_decode):
            continue
        segment_start = int(cumulative_tokens[req_idx]) - draft_count - 1
        rows = tuple(range(segment_start + 1, segment_start + draft_count + 1))
        if segment_start < 0 or rows[-1] >= scheduler_output.total_num_scheduled_tokens:
            raise ValueError("Activation-routing segment is outside the token buffer")
        row_tensor = torch.tensor(rows, dtype=torch.long, device="cpu")
        verify_rows.append((req_id, row_tensor))
        segments.append((segment_start, draft_count + 1))
    return tuple(verify_rows), segments


def build_verify_segments_v2(
    scheduler_output,
    req_ids,
    num_scheduled_tokens_np,
    num_computed_tokens_np,
    prefill_len_np,
    total_num_scheduled_tokens,
):
    """Build verify spans from MRv2's BatchReqState without V1-only fields."""
    verify_rows, segments = [], []
    draft_tokens = scheduler_output.scheduled_spec_decode_tokens
    if not draft_tokens:
        return tuple(verify_rows), segments
    cumulative_tokens = np.cumsum(num_scheduled_tokens_np, dtype=np.int32)
    req_index = {req_id: i for i, req_id in enumerate(req_ids)}
    for req_id, draft_token_ids in draft_tokens.items():
        req_idx = req_index.get(req_id)
        draft_count = len(draft_token_ids)
        if req_idx is None or not draft_count or num_computed_tokens_np[req_idx] < prefill_len_np[req_idx]:
            continue
        segment_start = int(cumulative_tokens[req_idx]) - draft_count - 1
        rows = tuple(range(segment_start + 1, segment_start + draft_count + 1))
        if segment_start < 0 or rows[-1] >= total_num_scheduled_tokens:
            raise ValueError("Activation-routing segment is outside the token buffer")
        verify_rows.append((req_id, torch.tensor(rows, dtype=torch.long)))
        segments.append((segment_start, draft_count + 1))
    return tuple(verify_rows), segments


def validate_activation_routing(
    state,
    model_config,
    speculative_config,
    uniform_decode_query_len,
    use_async_scheduling,
):
    """Fail fast when the runtime config violates the activation-routing contract."""
    uses_supported_drafter = (
        speculative_config is not None
        and (speculative_config.method == "dflash" or speculative_config.use_dspark())
    )
    if not uses_supported_drafter:
        raise ValueError("Activation routing requires DFlash or DSpark speculative decoding")
    if state.verify_block_size != uniform_decode_query_len:
        raise ValueError(
            "Activation-routing verify_block_size must equal the target "
            f"query length: {state.verify_block_size} != {uniform_decode_query_len}"
        )
    text_config = model_config.hf_text_config
    model_num_experts = getattr(
        text_config,
        "n_routed_experts",
        getattr(text_config, "num_experts", None),
    )
    if model_num_experts != state.num_experts:
        raise ValueError(
            "Activation-routing num_experts must match the model: "
            f"{state.num_experts} != {model_num_experts}"
        )
    model_route_top_k = getattr(text_config, "num_experts_per_tok", None)
    if model_route_top_k != state.route_top_k:
        raise ValueError(
            "Activation-routing route_top_k must match the model: "
            f"{state.route_top_k} != {model_route_top_k}"
        )
    model_scoring_func = getattr(text_config, "scoring_func", "softmax")
    if model_scoring_func != state.scoring_func:
        raise ValueError(
            "Activation-routing scoring_func must match the model: "
            f"{state.scoring_func} != {model_scoring_func}"
        )
    if use_async_scheduling:
        raise ValueError("Activation routing currently requires --no-async-scheduling")
    num_hash_layers = int(getattr(text_config, "num_hash_layers", 0) or 0)
    logger.info(
        "Activation routing enabled: backend=%s verify_block_size=%d "
        "num_experts=%d route_top_k=%d scoring_func=%s "
        "protected_rows=%d suffix_pool_top_k=%d skipped_hash_layers=%d",
        state.backend,
        state.verify_block_size,
        state.num_experts,
        state.route_top_k,
        state.scoring_func,
        state.protected_rows,
        state.suffix_pool_top_k,
        num_hash_layers,
    )


# Verified prefill fused-row bucket for the DSpark family. Kept as a family
# constant instead of binding max_num_batched_tokens: only row counts that
# were validated on hardware belong in the fused-route whitelist.
DSPARK_PREFILL_FUSED_ROWS = 512

# Speculative methods of the DSpark family (use_dspark() is method == "dspark").
DSPARK_SPECULATIVE_METHODS = ("dflash", "dspark")


def _derive_expected_router_layers(text_config):
    """Derive the target-router count from the model's hf text config.

    DSpark replaces the first ``num_hash_layers`` decoder blocks with hash
    routing (those blocks run no target router); every remaining block runs
    exactly one. Verified against DeepSeek-V4-Flash / -0731 config.json:
    num_hidden_layers=43, num_hash_layers=3 -> 40, matching the validated
    routing configuration. Returns None whenever the fields are missing or
    cannot yield a positive count: better no default than a wrong one.
    """
    if text_config is None:
        return None
    num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    num_hash_layers = getattr(text_config, "num_hash_layers", 0) or 0
    if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int):
        return None
    if num_hash_layers <= 0 or num_hidden_layers <= num_hash_layers:
        return None
    return num_hidden_layers - num_hash_layers


def _inject_dspark_defaults(
    routing_config,
    config_section,
    text_config,
    speculative_method,
    uniform_decode_query_len,
):
    """Fill DSpark-family defaults for fields the user did not write.

    Runs only for an enabled non-baseline config section under a DFlash /
    DSpark drafter. Explicitly written fields (``config_section.user_keys``)
    always keep their value; only unwritten fields get derived ones, and each
    derivation is logged once with its source.
    """
    if config_section.backend == "baseline":
        return
    if speculative_method not in DSPARK_SPECULATIVE_METHODS:
        return
    user_keys = getattr(config_section, "user_keys", frozenset())
    query_len_usable = (
        isinstance(uniform_decode_query_len, int)
        and not isinstance(uniform_decode_query_len, bool)
        and uniform_decode_query_len >= 2
    )
    derived = []
    if "verify_block_size" not in user_keys and query_len_usable:
        # validate_activation_routing already forces verify_block_size to
        # equal uniform_decode_query_len, so mirroring the engine truth is
        # the zero-risk default.
        routing_config["verify_block_size"] = uniform_decode_query_len
        derived.append(
            f"verify_block_size={uniform_decode_query_len} "
            "(derived from uniform_decode_query_len)"
        )
    if "fused_rows" not in user_keys and query_len_usable:
        block_size = routing_config["verify_block_size"]
        routing_config["fused_rows"] = [block_size, DSPARK_PREFILL_FUSED_ROWS]
        derived.append(
            f"fused_rows={routing_config['fused_rows']} "
            "(derived from verify_block_size and the verified prefill bucket)"
        )
    if "expected_router_layers" not in user_keys:
        expected = _derive_expected_router_layers(text_config)
        if expected is not None:
            routing_config["expected_router_layers"] = expected
            derived.append(
                f"expected_router_layers={expected} "
                "(derived from num_hidden_layers - num_hash_layers)"
            )
    if derived:
        logger.info(
            "[ACTIVATION-ROUTING] DSpark family defaults applied: %s", "; ".join(derived)
        )


def resolve_activation_routing_config(
    config_section,
    text_config=None,
    speculative_method=None,
    uniform_decode_query_len=None,
):
    """Resolve the activation-routing switch from additional_config.

    The ``additional_config.activation_routing`` section is the single
    configuration source. Under a DFlash / DSpark drafter, fields the user
    left unwritten get family defaults derived from the engine and model
    config (explicit values always win): ``verify_block_size`` from
    ``uniform_decode_query_len``, ``fused_rows`` from the verify block plus
    the verified prefill bucket, ``expected_router_layers`` from the hf
    text config. Other drafters keep the generic defaults untouched.
    A missing section (or the section disabled) -> None: the feature is
    fully off and every routing hook stays inert (baseline behavior).
    """
    if config_section is not None and config_section.enabled:
        routing_config = config_section.to_routing_config()
        _inject_dspark_defaults(
            routing_config,
            config_section,
            text_config,
            speculative_method,
            uniform_decode_query_len,
        )
        return DFlashTopMState.from_dict(routing_config, text_config=text_config)
    return None
