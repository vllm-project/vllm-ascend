# SPDX-License-Identifier: Apache-2.0
"""MRV1 batch adapter for the shared SFA DSpark draft-KV protocol."""

from types import SimpleNamespace
from typing import Any

from vllm.distributed.kv_transfer import get_kv_transfer_group

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    find_dspark_context_connector,
    find_dspark_prefix_connector,
    send_dspark_prefill_kv,
)


def send_mrv1_dspark_prefill_kv(runner: Any, scheduler_output: Any, aux_hidden_states: Any) -> None:
    """Use the reordered MRV1 batch's CPU mirrors before sampling advances it.

    Called after target forward has exited its attention context. The shared
    sender synchronizes draft writes before publishing their transfer notice.
    Connector finalization remains deferred until after the draft step.
    """
    connector, metadata = find_dspark_context_connector(get_kv_transfer_group(), scheduler_output.kv_connector_metadata)
    requests = getattr(metadata, "requests", {})
    if not any(getattr(req, "dspark_context_generation", None) for req in requests.values()):
        return
    if aux_hidden_states is None:
        raise RuntimeError("P produced no target auxiliary states for an active remote DSpark request")
    input_batch = runner.input_batch
    req_ids = input_batch.req_ids[: input_batch.num_reqs]
    batch = SimpleNamespace(
        num_reqs=len(req_ids),
        req_ids=req_ids,
        prefill_len_np=input_batch.num_prompt_tokens[: len(req_ids)],
        num_computed_tokens_np=input_batch.num_computed_tokens_cpu[: len(req_ids)],
        num_scheduled_tokens=[scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids],
        query_start_loc_np=runner.query_start_loc.np[: len(req_ids) + 1],
    )
    prefix_store = find_dspark_prefix_connector(get_kv_transfer_group(), scheduler_output.kv_connector_metadata)
    send_dspark_prefill_kv(
        runner.drafter,
        batch,
        aux_hidden_states,
        requests,
        connector,
        runner._dspark_prefill_progress,
        scheduler_output.finished_req_ids,
        prefix_connector=prefix_store[0] if prefix_store is not None else None,
        prefix_metadata=prefix_store[1] if prefix_store is not None else None,
    )
