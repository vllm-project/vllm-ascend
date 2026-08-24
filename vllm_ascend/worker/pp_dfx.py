# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Debug probes for Model Runner V2 pipeline-parallel transfers."""

import hashlib
import logging
import time
from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import logger

PP_DFX_METADATA_KEY = "__vllm_ascend_pp_dfx__"


class PPTransferDFX:
    """Add DEBUG probes around pipeline-parallel tensor transfers."""

    def __init__(self, use_v2_model_runner: bool) -> None:
        self.use_v2_model_runner = use_v2_model_runner
        self._send_seq = 0

    @property
    def enabled(self) -> bool:
        return self.use_v2_model_runner and logger.isEnabledFor(logging.DEBUG)

    def recv_tensor_dict(
        self,
        pp_group: Any,
        all_gather_group: Any,
    ) -> tuple[dict[str, torch.Tensor], list[Any], list[Callable[[], None]] | None]:
        """Receive tensors and defer verification until communication completes."""
        received_tensors, comm_handles, comm_postprocess = pp_group.irecv_tensor_dict(all_gather_group=all_gather_group)
        assert received_tensors is not None
        metadata = received_tensors.pop(PP_DFX_METADATA_KEY, None)
        if not self.enabled or metadata is None:
            return received_tensors, comm_handles, comm_postprocess

        transfer_seq = metadata["transfer_seq"]
        expected_fingerprints = metadata["fingerprints"]
        src_rank = (pp_group.rank_in_group - 1) % pp_group.world_size
        dst_rank = pp_group.rank_in_group

        def verify_received_tensors() -> None:
            verify_tensor_fingerprints(
                expected_fingerprints,
                received_tensors,
                transfer_seq,
                src_rank,
                dst_rank,
            )

        postprocess = list(comm_postprocess or [])
        # Verify the reconstructed payload after any TP all-gather.
        postprocess.append(verify_received_tensors)
        return received_tensors, comm_handles, postprocess

    def send_tensor_dict(
        self,
        pp_group: Any,
        tensors: dict[str, torch.Tensor],
        all_gather_group: Any,
    ) -> list[Any]:
        """Send tensors, measuring completion time when DEBUG logging is enabled."""
        if not self.enabled:
            return pp_group.isend_tensor_dict(
                tensors,
                all_gather_group=all_gather_group,
            )

        transfer_seq = self._send_seq
        self._send_seq += 1
        src_rank = pp_group.rank_in_group
        dst_rank = (src_rank + 1) % pp_group.world_size
        send_tensors = dict(tensors)
        send_tensors[PP_DFX_METADATA_KEY] = {
            "transfer_seq": transfer_seq,
            "fingerprints": compute_tensor_fingerprints(tensors),
        }

        # DEBUG DFX deliberately serializes the send so the wall-clock
        # duration includes HCCL completion instead of only submission.
        torch.npu.synchronize()
        start_time_ns = time.perf_counter_ns()
        send_work = pp_group.isend_tensor_dict(
            send_tensors,
            all_gather_group=all_gather_group,
        )
        for handle in send_work:
            handle.wait()
        torch.npu.synchronize()
        comm_elapsed_ms = (time.perf_counter_ns() - start_time_ns) / 1e6
        logger.debug(
            "[pp-dfx] seq=%d link=%d->%d comm=%.3fms",
            transfer_seq,
            src_rank,
            dst_rank,
            comm_elapsed_ms,
        )
        return []


def compute_tensor_fingerprints(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    """Compute bitwise fingerprints for a pipeline payload."""
    fingerprints: dict[str, str] = {}
    for name, tensor in tensors.items():
        digest = hashlib.sha256(usedforsecurity=False)
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        tensor_bytes = tensor.detach().contiguous().view(-1).cpu().view(torch.uint8)
        digest.update(tensor_bytes.numpy().tobytes())
        fingerprints[name] = digest.hexdigest()
    return fingerprints


def verify_tensor_fingerprints(
    expected_fingerprints: dict[str, str],
    tensors: dict[str, torch.Tensor],
    transfer_seq: int,
    src_rank: int,
    dst_rank: int,
) -> None:
    """Compare a received payload with its sender-side fingerprints."""
    received_fingerprints = compute_tensor_fingerprints(tensors)
    mismatch = next(
        (
            name
            for name in sorted(expected_fingerprints.keys() | received_fingerprints.keys())
            if expected_fingerprints.get(name) != received_fingerprints.get(name)
        ),
        None,
    )
    if mismatch is None:
        logger.debug(
            "[pp-dfx] seq=%d link=%d->%d verify=pass",
            transfer_seq,
            src_rank,
            dst_rank,
        )
    else:
        logger.debug(
            "[pp-dfx] seq=%d link=%d->%d verify=failed tensor=%s",
            transfer_seq,
            src_rank,
            dst_rank,
            mismatch,
        )
