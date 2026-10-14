# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary, test-owned text-divergence probes; remove after collecting the CI evidence.

Only CPU-side batch metadata is copied. No device tensors are read, no scheduling
decision is overridden, and no timing from an instrumented run is a benchmark.
"""

from typing import Any

import regex as re

from tests.e2e.pull_request.one_card.ascendstore_v1_performance import AscendStoreBenchmarkWorkerExtension

REQUEST_MARKER = "ascendstore-diag-"
MAX_TRACE_STEPS = 1024


class AscendStoreDiagnosticWorkerExtension(AscendStoreBenchmarkWorkerExtension):
    def start_ascendstore_diagnostic_trace(self) -> dict[str, bool]:
        assert not hasattr(self, "_diagnostic_prepare_inputs"), "A previous trace is still installed"
        runner = self.model_runner
        original = runner._prepare_inputs_impl
        self._diagnostic_prepare_inputs = original
        self._diagnostic_batches: list[dict[str, Any]] = []

        def traced_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            batch = original(*args, **kwargs)
            if any(REQUEST_MARKER in request_id for request_id in batch.req_ids):
                assert len(self._diagnostic_batches) < MAX_TRACE_STEPS, "Unexpected unbounded diagnostic trace"
                self._diagnostic_batches.append(snapshot_batch(batch, len(self._diagnostic_batches)))
            return batch

        # The wrapper belongs to this test's Worker instance, never the production class.
        runner._prepare_inputs_impl = traced_prepare_inputs
        return {"trace_installed": True}

    def stop_ascendstore_diagnostic_trace(self) -> dict[str, Any]:
        assert hasattr(self, "_diagnostic_prepare_inputs"), "No diagnostic trace is installed"
        self.model_runner._prepare_inputs_impl = self._diagnostic_prepare_inputs
        records = self._diagnostic_batches
        del self._diagnostic_prepare_inputs
        del self._diagnostic_batches
        return {"batches": records, "trace_restored": True}


def snapshot_batch(batch: Any, step: int) -> dict[str, Any]:
    """Preserve actual model-row order and copy mutable NumPy metadata before the next Step."""
    count = int(batch.num_reqs)
    return {
        "step": step,
        "request_ids": list(batch.req_ids),
        "num_reqs": count,
        "num_reqs_after_padding": int(batch.num_reqs_after_padding),
        "num_tokens": int(batch.num_tokens),
        "num_tokens_after_padding": int(batch.num_tokens_after_padding),
        "request_state_indices": batch.idx_mapping_np[:count].tolist(),
        "scheduled_tokens": batch.num_scheduled_tokens[:count].tolist(),
        "computed_tokens": batch.num_computed_tokens_np[:count].tolist(),
        "prompt_lengths": batch.prefill_len_np[:count].tolist(),
        "query_start_loc": batch.query_start_loc_np[: count + 1].tolist(),
        "is_prefilling": batch.is_prefilling_np[:count].tolist(),
        "attention_state": batch.attn_state.name,
    }


def request_index(request_id: str) -> int:
    match = re.search(r"ascendstore-diag-[a-z0-9-]+-p(\d+)(?:-|$)", request_id)
    assert match is not None, f"Unrecognized diagnostic request: {request_id}"
    return int(match.group(1))


def token_batch_context(batches: list[dict[str, Any]], index: int, token_position: int) -> dict[str, Any] | None:
    """Match a zero-based generated token to its actual input batch, with speculation disabled."""
    for batch in batches:
        for row, request_id in enumerate(batch["request_ids"]):
            if request_index(request_id) != index:
                continue
            generated_position = (
                batch["computed_tokens"][row] + batch["scheduled_tokens"][row] - batch["prompt_lengths"][row]
            )
            if generated_position == token_position:
                return {"model_row": row, **batch}
    return None


def first_divergence(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    """Compare the same prompt by token ID; never infer a token boundary by retokenizing text."""
    assert left["prompt_sha256"] == right["prompt_sha256"], "Mismatched prompts cannot establish output parity"
    left_ids, right_ids = left["token_ids"], right["token_ids"]
    assert len(left_ids) == len(right_ids), "Truncated responses must fail separately from text divergence"
    position = next((i for i, (a, b) in enumerate(zip(left_ids, right_ids)) if a != b), None)
    if position is None:
        return None
    result: dict[str, Any] = {"token_position": position, "shared_prefix_token_ids": left_ids[:position]}
    for name, observation in (("left", left), ("right", right)):
        logprobs = observation.get("logprobs")
        candidates = None if logprobs is None else logprobs["top_logprobs"][position]
        ordered = [] if not candidates else sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        result[name] = {
            "token_id": observation["token_ids"][position],
            "top_logprobs": candidates,
            # A small gap supports numerical sensitivity; it does not by itself prove correct KV transfer.
            "top_two_logprob_gap": ordered[0][1] - ordered[1][1] if len(ordered) >= 2 else None,
        }
    return result


def compare_phases(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep repeat, batch/order, connector, Load and instrumentation contrasts separate."""
    comparisons = []
    by_key = {(phase["implementation"], phase["case"], phase["source"], phase["repeat"]): phase for phase in phases}
    for right in phases:
        implementation, case, source, repeat = (right[key] for key in ("implementation", "case", "source", "repeat"))
        references = []
        if repeat == 2:
            references.append(("repeat", (implementation, case, source, 1)))
        if implementation != "none":
            references.append(("connector", ("none", case, source, repeat)))
        if implementation == "v1":
            references.append(("production-v1", ("production", case, source, repeat)))
        if source == "external":
            references.append(("cold-external", (implementation, case, "cold", repeat)))
        elif case.startswith("c8"):
            references.append(("batch-size", (implementation, "c1", source, repeat)))
            if case != "c8-natural":
                references.append(("submission-order", (implementation, "c8-natural", source, repeat)))
        elif case.startswith("control"):
            references.append(("instrumentation", (implementation, "c8-natural", source, 1)))
        for contrast, key in references:
            left = by_key.get(key)
            if left is None:
                continue
            differences = []
            assert len(left["observations"]) == len(right["observations"])
            for a, b in zip(left["observations"], right["observations"], strict=True):
                assert a["request_index"] == b["request_index"]
                difference = first_divergence(a, b)
                if difference is not None:
                    index, position = a["request_index"], difference["token_position"]
                    differences.append({"request_index": index, **difference})
                    for side, phase in (("left", left), ("right", right)):
                        difference[side]["batch"] = token_batch_context(phase["batches"], index, position)
            comparisons.append(
                {"contrast": contrast, "left": left["id"], "right": right["id"], "differences": differences}
            )
    return comparisons
