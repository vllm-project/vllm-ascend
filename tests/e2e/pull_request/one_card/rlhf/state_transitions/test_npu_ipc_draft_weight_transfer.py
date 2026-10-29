# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real NPU IPC transfers into a standalone draft, followed by a target update."""

import gc

import pytest
import torch
import torch_npu  # noqa: F401
from transformers import Qwen3Config
from vllm import LLM, SamplingParams
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource

from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerInitInfo,
    NPUIPCTrainerWeightTransferEngine,
)

WEIGHT_NAME = "model.norm.weight"
HIDDEN_SIZE = 256


class _NormWeightSource(WeightSource):
    def __init__(self):
        self.value = 2.0

    def metadata(self):
        return [ParamMeta(WEIGHT_NAME, torch.bfloat16, (HIDDEN_SIZE,))]

    def __iter__(self):
        yield WEIGHT_NAME, torch.full((HIDDEN_SIZE,), self.value, dtype=torch.bfloat16, device="npu:0")


class _LocalClient:
    def __init__(self, llm):
        self.llm = llm
        self.is_draft = True

    def init_weight_transfer_engine(self, init_info):
        self.llm.init_weight_transfer_engine({"init_info": init_info})

    def start_weight_update(self):
        if self.is_draft:
            self.llm.start_draft_weight_update()
        else:
            self.llm.start_weight_update()

    def update_weights(self, update_info):
        self.llm.update_weights({"update_info": update_info})

    def finish_weight_update(self, weight_version=None):
        self.llm.finish_weight_update(weight_version)


def _snapshot(worker):
    result = []
    for model in (worker.get_model(), worker.get_draft_model()):
        weight = model.get_parameter(WEIGHT_NAME)
        result.append((weight.detach().cpu(), weight.data_ptr()))
    return result


@pytest.mark.skipif(torch.npu.device_count() < 1, reason="Requires one NPU for colocated IPC transfer.")
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_npu_ipc_draft_then_target_update(tmp_path, monkeypatch, packed):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # The trusted local RPCs carry a callable snapshot and IPC rebuild arguments.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    torch.npu.set_device(0)
    model_path = tmp_path / "tiny-qwen3"
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=128,
        bos_token_id=1,
        eos_token_id=127,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    ).save_pretrained(model_path)
    llm = LLM(
        model=str(model_path),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.2,
        speculative_config={"method": "standalone_ar", "model": str(model_path), "num_speculative_tokens": 2},
        weight_transfer_config={"backend": "npu_ipc"},
        additional_config={"rl_config": {"enabled": True}},
    )
    trainer = None
    try:
        before = llm.collective_rpc(_snapshot)[0]
        client = _LocalClient(llm)
        source = _NormWeightSource()
        trainer = NPUIPCTrainerWeightTransferEngine.trainer_init(
            NPUIPCTrainerInitInfo(rank=0, packed=packed, packed_buffer_size_bytes=4096),
            client=client,
            source=source,
        )
        # Repeated draft sessions must leave the target model unchanged.
        for value in (2.0, 3.0):
            source.value = value
            trainer.send_weights()
            target, draft = llm.collective_rpc(_snapshot)[0]
            torch.testing.assert_close(target[0], before[0][0], rtol=0, atol=0)
            torch.testing.assert_close(draft[0], torch.full_like(draft[0], value), rtol=0, atol=0)
            assert (target[1], draft[1]) == (before[0][1], before[1][1])

        client.is_draft = False
        source.value = 5.0
        trainer.send_weights()
        target, draft = llm.collective_rpc(_snapshot)[0]
        torch.testing.assert_close(target[0], torch.full_like(target[0], 5.0), rtol=0, atol=0)
        torch.testing.assert_close(draft[0], torch.full_like(draft[0], 3.0), rtol=0, atol=0)
        assert (target[1], draft[1]) == (before[0][1], before[1][1])
        output = llm.generate(
            [{"prompt_token_ids": [1, 2, 3]}], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        )
        assert len(output[0].outputs[0].token_ids) == 4
    finally:
        if trainer is not None:
            trainer.shutdown()
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.npu.empty_cache()
