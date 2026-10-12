# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np
import torch
import torch_npu  # noqa: F401

from vllm_ascend.worker.v2.input_staging import BatchInputStaging


def test_staged_inputs_keep_query_address_and_order_all_device_consumers():
    device = torch.device("npu:0")
    max_reqs = 16
    staging = BatchInputStaging(max_reqs, device)
    query_out = torch.empty(max_reqs + 2, dtype=torch.int32, device=device)
    query_pointer = query_out.data_ptr()
    results, expected = [], []
    for step in range(80):
        count = step % max_reqs + 1
        mapping = np.arange(count, dtype=np.intp) + step
        logits = np.arange(count + 1, dtype=np.int32) * 8
        query = np.full(max_reqs + 2, count * 8, dtype=np.int32)
        query[: count + 1] = logits
        staged = staging.copy(mapping, logits, query, query_out)
        if staged is None:
            # The reference path owns fresh sources; no staging bank is touched.
            idx = torch.tensor(mapping, device=device)
            cu = torch.tensor(logits, device=device)
            query_out.copy_(torch.from_numpy(query))
        else:
            idx, cu = staged
            assert idx.dtype == torch.int64
            assert cu.dtype == torch.int32
        # Queue consumers before another iteration can overwrite device inputs.
        results.append(torch.cat((idx, cu, query_out)).clone())
        expected.append(np.concatenate((mapping, logits, query)))
    torch.npu.synchronize()
    for actual, reference in zip(results, expected):
        np.testing.assert_array_equal(actual.cpu().numpy(), reference)
    assert query_out.data_ptr() == query_pointer
    assert len(staging.banks) <= 2
    assert all(host.is_pinned() for host, _, _ in staging.banks)
    with torch.npu.stream(torch.npu.Stream()):
        assert staging.copy(mapping, logits, query, query_out) is None
