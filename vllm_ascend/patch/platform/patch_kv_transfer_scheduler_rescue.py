#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorOutput
from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus


def _patched_update_from_kv_xfer_finished(self: Scheduler, kv_connector_output: KVConnectorOutput):
    """
    Keep vLLM logic and replace missing-request assert with warning+skip.
    """

    if self.connector is not None:
        self.connector.update_connector_output(kv_connector_output)

    for req_id in kv_connector_output.finished_recving or ():
        logger.debug("Finished recving KV transfer for request %s", req_id)
        if req_id not in self.requests:
            logger.warning(
                "[KV-XFER-RESCUED] req_id %s not in requests during finished_recving, likely aborted, skipping.",
                req_id,
            )
            continue
        req = self.requests[req_id]
        if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            self.finished_recving_kv_req_ids.add(req_id)
        else:
            assert RequestStatus.is_finished(req.status)
            self._free_blocks(self.requests[req_id])

    for req_id in kv_connector_output.finished_sending or ():
        logger.debug("Finished sending KV transfer for request %s", req_id)
        if req_id not in self.requests:
            logger.warning(
                "[KV-XFER-RESCUED] req_id %s not in requests during finished_sending, likely aborted, skipping.",
                req_id,
            )
            continue
        self._free_blocks(self.requests[req_id])


Scheduler._update_from_kv_xfer_finished = _patched_update_from_kv_xfer_finished  # type: ignore[method-assign]
