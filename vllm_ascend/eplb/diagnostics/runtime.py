# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, per-layer workload logging across one EP group."""

import functools
import logging
from collections import Counter
from types import SimpleNamespace

import torch

from vllm_ascend.eplb.diagnostics.decision import EnablementDecision, make_planner
from vllm_ascend.eplb.diagnostics.probe import ExpertLoadProbe

_LOG_NAME = "vllm.eplb.diagnostics"


def layer_work(rows, window_size):
    """Validate source conservation and map valid routes to actual expert owners."""
    if len({r["step"] for r in rows}) != 1:
        return None, "unaligned_windows"
    names = [x["name"] for x in rows[0]["layers"]]
    if len(names) != 1 or any([x["name"] for x in r["layers"]] != names for r in rows):
        return None, "incomplete_layer_set"
    ranks = [r["rank"] for r in rows]
    if len(set(ranks)) != len(ranks):
        return None, "duplicate_rank"
    sources = {}
    for row in rows:
        sources.setdefault(tuple(row["tp_ranks"]), []).append(row)
    for tp, members in sources.items():
        if set(tp) != {r["rank"] for r in members} or len({r["tokens"] for r in members}) != 1:
            return None, "source_tp_mismatch"
    rank_work, hot = [0] * len(rows), {}
    for index, name in enumerate(names):
        layers = [r["layers"][index] for r in rows]
        n = len(layers[0]["counts"]) - 2
        top_k = layers[0]["top_k"]
        if n <= 0 or top_k <= 0 or any(len(x["counts"]) != n + 2 or x["top_k"] != top_k for x in layers):
            return None, "expert_space_mismatch"
        if any(x["counts"][-2] != window_size or x["counts"][-1] or min(x["counts"]) < 0 for x in layers):
            return None, "unsupported_routing_or_call_count"
        for members in sources.values():
            actual = sum(sum(r["layers"][index]["counts"][:-2]) for r in members)
            if actual != members[0]["tokens"] * top_k:
                return None, "source_conservation_failure"
        owners = {}
        for owner, layer in enumerate(layers):
            mapping = layer["mapping"]
            local_slots = [slot for slot in mapping if slot >= 0]
            if (
                len(mapping) != n
                or len(local_slots) != layer["local_experts"]
                or set(local_slots) != set(range(layer["local_experts"]))
            ):
                return None, "invalid_expert_mapping"
            for expert, slot in enumerate(mapping):
                if slot >= 0:
                    if expert in owners:
                        return None, "duplicate_expert_owner"
                    owners[expert] = owner
        if len(owners) != n:
            return None, "missing_expert_owner"
        loads = [sum(x["counts"][e] for x in layers) for e in range(n)]
        for expert, count in enumerate(loads):
            rank_work[owners[expert]] += count
        mean = sum(loads) / n
        cutoff = sorted(loads, reverse=True)[min(4, n) - 1]
        hot.update(
            {(name, e): count for e, count in enumerate(loads) if count > 0 and count >= max(cutoff, 1.2 * mean)}
        )
    total = sum(rank_work)
    if not total:
        return None, "no_real_work"
    history = None
    if any("history" in layer for layer in layers):
        histories = [layer.get("history", []) for layer in layers]
        if any(
            len(h) != window_size
            or any(len(step) != n + 2 or step[-2] != 1 or step[-1] or min(step) < 0 for step in h)
            or [sum(step[e] for step in h) for e in range(n + 2)] != layer["counts"]
            for h, layer in zip(histories, layers)
        ):
            return None, "invalid_step_history"
        for members in sources.values():
            expected = members[0].get("token_steps", [])
            if len(expected) != window_size or any(row.get("token_steps") != expected for row in members):
                return None, "unaligned_step_tokens"
            if any(
                tokens < 0 or sum(sum(row["layers"][0]["history"][step][:-2]) for row in members) != tokens * top_k
                for step, tokens in enumerate(expected)
            ):
                return None, "step_source_conservation_failure"
        history = [[sum(h[step][e] for h in histories) for e in range(n)] for step in range(window_size)]
    return {
        "history": history,
        "rank_work": rank_work,
        "total": total,
        "skew": max(rank_work) * len(rows) / total,
        "hot": hot,
        "expert_work": [{e: loads[e] for e in range(n) if owners[e] == rank} for rank in range(len(rows))],
    }, None


class LayerLogger:
    """Log cumulative work and a bounded consecutive-window enablement decision."""

    def __init__(self, ranks, stage, layer, planner=None, policy="unavailable"):
        self.ranks, self.stage, self.layer = ranks, stage, layer
        self.valid_windows = self.invalid_windows = self.observed_calls = 0
        self.expert_totals = [Counter() for _ in ranks]
        self.decision = EnablementDecision(planner, policy)
        self.hint = "insufficient_evidence"

    def log(self, rows, window, window_size, complete=True):
        result, reason = layer_work(rows, window_size)
        logger = logging.getLogger(_LOG_NAME)
        if reason:
            self.invalid_windows += 1
            self.decision.reset(reason)
            self.hint = "insufficient_evidence"
            logger.info(
                "[EPLB diagnostic] stage=%s layer=%s window=%s insufficient_evidence=%s",
                self.stage,
                self.layer,
                window,
                reason,
            )
            return
        self.observed_calls += window_size
        self.valid_windows += int(complete)
        self.decision.observe(result, rows, window, complete)
        self.hint = self.decision.verdict
        for rank, counts, total, work in zip(
            self.ranks, result["expert_work"], self.expert_totals, result["rank_work"]
        ):
            total.update(counts)
            logger.info(
                "[EPLB experts] stage=%s layer=%s rank=%s window=%s window_work=%s "
                "cumulative_work=%s window_expert_work=%s expert_work=%s",
                self.stage,
                self.layer,
                rank,
                window,
                work,
                sum(total.values()),
                counts,
                [f"{e}:{count}" for e, count in sorted(total.items())],
            )
        cumulative = [sum(counts.values()) for counts in self.expert_totals]
        peak = max(cumulative)
        logger.info(
            "[EPLB diagnostic] stage=%s layer=%s window=%s calls=%s ranks=%s "
            "valid_assignments=%s rank_work=%s window_rank_max_mean=%.3f "
            "cumulative_rank_max_mean=%.3f busiest_rank=%s hot_experts=%s "
            "evidence=%s hint=%s phases=%s timing_evidence=not_collected net_benefit=unknown",
            self.stage,
            self.layer,
            window,
            window_size,
            self.ranks,
            result["total"],
            result["rank_work"],
            result["skew"],
            peak * len(self.ranks) / sum(cumulative),
            self.ranks[cumulative.index(peak)],
            sorted(result["hot"]),
            self.decision.report(),
            self.hint,
            sorted({phase for row in rows for phase, _ in row["phases"]}),
        )


class WorkloadLogger:
    """Print each rank/layer, then an overall assessment based on layer evidence."""

    def __init__(self, ranks, stage, planner=None, policy="unavailable"):
        self.ranks, self.stage = ranks, stage
        self.layers = {}
        self.planner, self.policy = planner, policy

    def log(self, rows, window, window_size, complete=True):
        names = sorted({layer["name"] for row in rows for layer in row["layers"]} | self.layers.keys())
        for name in names:
            if name not in self.layers:
                self.layers[name] = LayerLogger(self.ranks, self.stage, name, self.planner, self.policy)
            scoped = [dict(row, layers=[layer for layer in row["layers"] if layer["name"] == name]) for row in rows]
            self.layers[name].log(scoped, window, window_size, complete)
        self.summarize(window)

    def summarize(self, window, final=False):
        names = list(self.layers)
        counts = Counter(layer.hint for layer in self.layers.values())
        candidates = [name for name, layer in self.layers.items() if layer.hint == "recommend_trial"]
        conclusion = "insufficient_evidence"
        if candidates:
            conclusion = "recommend_trial"
        elif counts["not_recommended_now"] == len(names) and names:
            conclusion = "not_recommended_now"
        logging.getLogger(_LOG_NAME).info(
            "[EPLB summary] stage=%s window=%s final=%s ranks=%s layers=%s layer_assessments=%s candidate_layers=%s "
            "invalid_layer_windows=%s conclusion=%s scope=workload_screening "
            "timing_evidence=not_collected net_benefit=unknown speedup=not_estimated",
            self.stage,
            window,
            final,
            self.ranks,
            len(names),
            dict(counts),
            candidates,
            sum(layer.invalid_windows for layer in self.layers.values()),
            conclusion,
        )


class DiagnosticsRecorder:
    def __init__(self, config, layers, group, tp_ranks, stage, source_tokens, planner=None, policy="unavailable"):
        self.config, self.layers, self.group = config, layers, group
        self.tp_ranks, self.source_tokens = list(tp_ranks), source_tokens
        self.summary = WorkloadLogger(list(group.ranks), stage, planner, policy)
        self.step = self.tokens = 0
        self.phases = set()
        self.token_steps = []
        self.history_slots = list(
            {
                id(layer.eplb_diagnostic_probe.history_slot): layer.eplb_diagnostic_probe.history_slot
                for _, layer in layers
                if layer.eplb_diagnostic_probe.history_slot is not None
            }.values()
        )
        self.collecting = self.real = self.finished = False
        self.layout = [
            {
                "name": name,
                "mapping": layer.ascend_expert_map.cpu().tolist(),
                "top_k": layer.moe_config.experts_per_token,
                "local_experts": layer.local_num_experts,
            }
            for name, layer in layers
        ]

    def begin(self, tokens, dummy=False):
        self.step += 1
        offset = self.step - self.config.warmup_steps - 1
        self.collecting = (
            not self.finished
            and offset >= 0
            and (self.config.max_windows == 0 or offset < self.config.window_size * self.config.max_windows)
        )
        self.real = not dummy
        if self.collecting and offset % self.config.window_size == 0:
            self.tokens = 0
            self.phases.clear()
            self.token_steps = []
            for _, layer in self.layers:
                layer.eplb_diagnostic_probe.totals.zero_()
                if layer.eplb_diagnostic_probe.history is not None:
                    layer.eplb_diagnostic_probe.history.zero_()
        self.source_tokens.fill_(tokens if self.collecting and self.real else 0)
        if self.collecting and self.real:
            self.tokens += tokens
        if self.collecting:
            self.token_steps.append(tokens if self.real else 0)
            for slot in self.history_slots:
                slot.fill_(offset % self.config.window_size)

    def end(self):
        if not self.collecting or (self.step - self.config.warmup_steps) % self.config.window_size:
            return
        self._collect(self.config.window_size)

    def _collect(self, calls, complete=True):
        # Deliberately synchronous only once per window. All group members,
        # including idle DP participants, must join in model-collective order.
        # Dummy requests contribute no routes/tokens or timing; their participation
        # is control bookkeeping. No world/PP collective is added here.
        probes = [layer.eplb_diagnostic_probe for _, layer in self.layers]
        counters = [probe.history[:calls] if probe.history is not None else probe.totals for probe in probes]
        snapshots = torch.cat([c.flatten() for c in counters]).cpu().split([counter.numel() for counter in counters])
        layer_rows = []
        for layout, snapshot, probe in zip(self.layout, snapshots, probes):
            if probe.history is None:
                layer_rows.append(dict(layout, counts=snapshot.tolist()))
            else:
                history = snapshot.reshape(calls, -1)
                layer_rows.append(dict(layout, counts=history.sum(dim=0).tolist(), history=history.tolist()))
        row = {
            "step": self.step,
            "rank": self.group.ranks[self.group.rank_in_group],
            "tp_ranks": self.tp_ranks,
            "tokens": self.tokens,
            "token_steps": self.token_steps,
            "phases": sorted(self.phases),
            "layers": layer_rows,
        }
        rows = [None] * len(self.group.ranks)
        # Gloo writes its temporary receive tensors from a background thread.
        with torch.inference_mode(False):
            torch.distributed.all_gather_object(rows, row, group=self.group.cpu_group)
        if self.group.rank_in_group == 0:
            self.summary.log(
                rows, (self.step - self.config.warmup_steps - 1) // self.config.window_size + 1, calls, complete
            )

    @torch.inference_mode()
    def finish(self):
        """Called collectively after generation, never from asynchronous teardown."""
        if self.finished:
            return
        pending = (self.step - self.config.warmup_steps) % self.config.window_size
        if self.collecting and pending:
            self._collect(pending, complete=False)
        self.finished = True
        self.source_tokens.zero_()
        if self.group.rank_in_group == 0:
            self.summary.summarize(
                max(0, (self.step - self.config.warmup_steps - 1) // self.config.window_size + 1), final=True
            )


def annotate_batch(runner, **metadata):
    recorder = getattr(runner, "_eplb_diagnostics_recorder", None)
    if recorder is not None and recorder.collecting and recorder.real:
        recorder.phases.add((metadata["phase"], metadata["graph_mode"]))


@torch.inference_mode()
def initialize_diagnostics(runner):
    if runner.ascend_config.eplb_diagnostics.mode == "off":
        return
    parallel = runner.vllm_config.parallel_config
    if not parallel.enable_expert_parallel or parallel.enable_eplb or runner.ascend_config.eplb_config.dynamic_eplb:
        raise ValueError("EPLB diagnostics requires EP enabled and EPLB disabled.")
    config = runner.vllm_config
    if getattr(parallel, "enable_elastic_ep", False):
        raise ValueError("EPLB diagnostics requires fixed EP membership.")
    capacity = max(
        config.scheduler_config.max_num_batched_tokens, max(config.compilation_config.cudagraph_capture_sizes or [0])
    )
    source_tokens = torch.zeros((), dtype=torch.int64, device=runner.device)
    positions = torch.arange(capacity, device=runner.device)
    history_slot = torch.zeros(1, dtype=torch.int64, device=runner.device)
    for layer in runner.model.modules():
        probe = getattr(layer, "eplb_diagnostic_probe", None)
        if isinstance(probe, ExpertLoadProbe):
            probe.source_token_count, probe.source_positions = source_tokens, positions
            probe.history = torch.zeros(
                (runner.ascend_config.eplb_diagnostics.window_size, probe.num_experts + 2),
                dtype=torch.int64,
                device=runner.device,
            )
            probe.history_slot = history_slot
    runner._eplb_diagnostics_source_tokens = source_tokens


def start_diagnostics(runner):
    if runner.ascend_config.eplb_diagnostics.mode == "off":
        return
    # Import after worker initialization; portable tests do not initialize vLLM.
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    from vllm_ascend.distributed.parallel_state import get_mc2_group

    layers = [
        (getattr(layer, "layer_name", name), layer)
        for name, layer in runner.model.named_modules()
        if isinstance(getattr(layer, "eplb_diagnostic_probe", None), ExpertLoadProbe)
    ]
    if not layers or any(
        layer.ascend_expert_map is None or layer.moe_config.num_experts != layer.moe_config.num_logical_experts
        for _, layer in layers
    ):
        raise ValueError("EPLB diagnostics requires explicit expert ownership without redundant experts.")
    group = get_mc2_group()
    planner, policy = (None, "non_reporting_rank")
    if group.rank_in_group == 0:
        planner, policy = make_planner(runner, len(group.ranks))
    runner._eplb_diagnostics_recorder = DiagnosticsRecorder(
        runner.ascend_config.eplb_diagnostics,
        layers,
        get_mc2_group(),
        get_tp_group().ranks,
        get_pp_group().rank_in_group,
        runner._eplb_diagnostics_source_tokens,
        planner,
        policy,
    )
    logging.getLogger(_LOG_NAME).info(
        "EPLB diagnostics: per-layer valid MoE assignments across each EP group/stage; synchronous window summaries; "
        "dummy/padding excluded. recommend_trial is workload evidence, not a measured speedup."
    )


def record_diagnostics(func):
    @functools.wraps(func)
    def wrapped(runner, scheduler_output, *args, **kwargs):
        recorder = getattr(runner, "_eplb_diagnostics_recorder", None)
        dummy = kwargs.get("dummy_run", False)
        if (
            recorder is None
            or getattr(runner, "_eplb_diagnostics_in_execution", False)
            or kwargs.get("is_profile", False)
            or torch.npu.is_current_stream_capturing()
            or (scheduler_output.total_num_scheduled_tokens == 0 and not dummy)
        ):
            return func(runner, scheduler_output, *args, **kwargs)
        runner._eplb_diagnostics_in_execution = True
        try:
            with torch.inference_mode():
                recorder.begin(scheduler_output.total_num_scheduled_tokens, dummy)
            result = func(runner, scheduler_output, *args, **kwargs)
            recorder.end()
            return result
        finally:
            runner._eplb_diagnostics_in_execution = False

    return wrapped


@record_diagnostics
def _record_dummy_batch(runner, scheduler_output, num_tokens, *, dummy_run):
    return runner._dummy_run(num_tokens, uniform_decode=True)


def run_dummy_batch(runner, num_tokens):
    metadata = SimpleNamespace(total_num_scheduled_tokens=0)
    return _record_dummy_batch(runner, metadata, num_tokens, dummy_run=True)
