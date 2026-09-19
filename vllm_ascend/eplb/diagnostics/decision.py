# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workload screening with EPLB off. Candidate layouts never change the model."""

import torch

# Screening heuristics, not performance thresholds or a speedup guarantee.
MIN_WINDOWS = 3
MIN_RANK_MAX_MEAN = 1.2
MIN_PEAK_REDUCTION = 0.05
MIN_PASS_FRACTION = 0.8
MIN_HOT_OVERLAP = 0.5


def make_planner(runner, ranks):
    """Use the configured production policy on CPU copies, or report a gap."""
    parallel = runner.vllm_config.parallel_config
    if runner.vllm_config.use_v2_model_runner:
        from vllm.distributed.eplb.policy import EPLB_POLICIES
        from vllm.distributed.parallel_state import get_node_count

        policy = EPLB_POLICIES.get(parallel.eplb_config.policy)
        groups = getattr(runner.model, "num_expert_groups", None)
        nodes = get_node_count()
        nodes = nodes if ranks % nodes == 0 else 1
        if policy is None or not isinstance(groups, int) or groups < 1:
            return None, "unsupported_mrv2_policy_or_expert_groups"

        def plan(source, loads):
            experts = len(loads)
            return (
                policy.rebalance_experts(
                    torch.tensor([loads], dtype=torch.float32, device="cpu"),
                    experts,
                    groups,
                    nodes,
                    ranks,
                    torch.tensor(source, device="cpu").reshape(1, experts),
                )
                .reshape(ranks, -1)
                .tolist()
            )

        return plan, "mrv2:" + parallel.eplb_config.policy

    policy_type = runner.ascend_config.eplb_config.eplb_policy_type
    if policy_type == 1:
        from vllm_ascend.eplb.core.policy.policy_default_eplb import DefaultEplb

        policy = DefaultEplb()
    elif policy_type == 2:
        from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb

        policy = SwiftBalanceEplb()
    else:
        # Random and multistage policies cannot be inferred from a single layer.
        return None, "unsupported_mrv1_policy"

    def plan(source, loads):
        work = torch.tensor([[[loads[e] for e in rank] for rank in source]], dtype=torch.float32, device="cpu")
        _, _, candidate = policy.rebalance_experts(torch.tensor([source], device="cpu"), work)
        return candidate[0]

    return plan, f"mrv1:{policy_type}"


def peak_work(history, layout):
    """Sum per-step rank maxima, so a changing busiest rank cannot cancel out."""
    return sum(max(sum(step[e] for e in rank) for rank in layout) for step in history)


class EnablementDecision:
    """Bounded consecutive-window evidence, evaluated before the next plan is built."""

    def __init__(self, planner=None, policy="unavailable"):
        self.planner, self.policy = planner, policy
        self.reset("need_complete_windows")

    def reset(self, reason):
        self.previous = self.signature = self.last_window = None
        self.windows = self.pairs = self.imbalanced = self.improved = self.stable = 0
        self.baseline_peak = self.candidate_peak = 0
        self.last_reduction = self.last_overlap = None
        self.last_skew = None
        self.verdict, self.reason = "insufficient_evidence", reason

    def observe(self, result, rows, window, complete):
        if not complete:
            self.previous = None
            return
        signature = tuple((r["rank"], tuple(r["layers"][0]["mapping"])) for r in rows)
        # Phase labels describe the workload; mixed batches and phase changes
        # remain valid evidence. Held-out work tests whether a plan generalizes.
        if signature != self.signature or (self.last_window is not None and window != self.last_window + 1):
            self.reset("need_complete_windows")
        self.signature, self.last_window = signature, window
        history = result.get("history")
        if history is None:
            self.reset("missing_step_history")
            return
        source = tuple(
            tuple(sorted(work, key=lambda e: row["layers"][0]["mapping"][e]))
            for row, work in zip(rows, result["expert_work"])
        )
        if len(source) < 2 or len({len(rank) for rank in source}) != 1:
            self.reset("unsupported_expert_capacity")
            return
        baseline = peak_work(history, source)
        self.windows += 1
        self.last_skew = baseline * len(source) / result["total"]
        self.imbalanced += self.last_skew >= MIN_RANK_MAX_MEAN
        hot = set(result["hot"])
        if self.previous is not None:
            candidate, previous_hot = self.previous
            predicted = peak_work(history, candidate)
            self.last_reduction = 1 - predicted / baseline
            self.last_overlap = len(hot & previous_hot) / max(1, len(hot | previous_hot))
            self.pairs += 1
            self.baseline_peak += baseline
            self.candidate_peak += predicted
            self.improved += self.last_reduction >= MIN_PEAK_REDUCTION
            self.stable += bool(hot) and self.last_overlap >= MIN_HOT_OVERLAP
        self._decide()
        if self.planner is None:
            self.verdict, self.reason = "insufficient_evidence", self.policy
            return
        loads = [sum(step[e] for step in history) for e in range(len(history[0]))]
        try:
            # Only CPU snapshots enter policy code. Validate every returned slot.
            candidate = self.planner(source, loads)
            candidate = tuple(tuple(int(e) for e in rank) for rank in candidate)
            if (
                len(candidate) != len(source)
                or any(len(new) != len(old) for new, old in zip(candidate, source))
                or sorted(e for rank in candidate for e in rank) != list(range(len(loads)))
            ):
                raise ValueError("invalid candidate")
        except Exception:
            self.reset("planner_failed_or_invalid_candidate")
            return
        self.previous = candidate, hot

    def _decide(self):
        self.verdict, self.reason = "insufficient_evidence", "need_complete_windows"
        if self.windows < MIN_WINDOWS or self.pairs < MIN_WINDOWS - 1:
            return
        if self.imbalanced / self.windows < MIN_PASS_FRACTION:
            self.verdict, self.reason = "not_recommended_now", "no_persistent_rank_imbalance"
        elif self.stable / self.pairs < MIN_PASS_FRACTION:
            self.reason = "hotspots_not_predictable"
        elif (
            self.improved / self.pairs < MIN_PASS_FRACTION
            or 1 - self.candidate_peak / self.baseline_peak < MIN_PEAK_REDUCTION
        ):
            self.verdict, self.reason = "not_recommended_now", "candidate_did_not_generalize"
        else:
            self.verdict, self.reason = "recommend_trial", "persistent_imbalance_and_future_work_reduction"

    def report(self):
        return dict(
            decision=self.verdict,
            reason=self.reason,
            policy=self.policy,
            complete_windows=self.windows,
            evaluated_pairs=self.pairs,
            imbalanced_windows=self.imbalanced,
            improved_pairs=self.improved,
            stable_hotspot_pairs=self.stable,
            last_hotspot_jaccard=self.last_overlap,
            step_peak_over_mean=self.last_skew,
            heldout_baseline_peak_work=self.baseline_peak,
            heldout_candidate_peak_work=self.candidate_peak,
            last_peak_work_reduction=self.last_reduction,
            heldout_peak_work_reduction=(1 - self.candidate_peak / self.baseline_peak if self.baseline_peak else None),
        )
