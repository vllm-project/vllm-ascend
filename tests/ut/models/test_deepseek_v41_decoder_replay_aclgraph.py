# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_ascend.models.deepseek_v41.decoder_replay_aclgraph import DecoderReplayAclGraphManager


def test_replay_graph_reuses_static_inputs():
    manager = object.__new__(DecoderReplayAclGraphManager)
    manager.state = SimpleNamespace(
        config=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1)),
        layers=SimpleNamespace(sequence_parallel=False),
    )
    manager.input_buffers = tuple(torch.zeros(8, dtype=torch.float32) for _ in range(4))
    addresses = []

    def run(*inputs):
        addresses.append(tuple(value.data_ptr() for value in inputs))
        return (inputs[0].clone(),)

    manager.runner = run
    first = manager.run(torch.arange(4), tuple(torch.arange(4, dtype=torch.float32) for _ in range(4)))
    second = manager.run(torch.arange(4), tuple(torch.arange(4, dtype=torch.float32) + 10 for _ in range(4)))

    assert first[0].tolist() == [0, 1, 2, 3]
    assert second[0].tolist() == [10, 11, 12, 13]
    assert addresses[0] == addresses[1]
