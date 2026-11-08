# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded reloads must match independent cold loading of a dense Qwen2."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401


def _snapshot(worker, directory, label):
    torch.npu.synchronize()
    parameters = dict(worker.get_model().named_parameters())
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in parameters.items()},
        Path(directory) / f"{label}-parameters.pt",
    )
    return {name: tensor.data_ptr() for name, tensor in parameters.items()}


class _WorkerClient:
    def __init__(self, llm):
        self.llm = llm

    def init_weight_transfer_engine(self, init_info):
        self.llm.collective_rpc("init_weight_transfer_engine", timeout=120, args=(init_info,))

    def start_weight_update(self):
        self.llm.collective_rpc("start_weight_update", timeout=120)

    def update_weights(self, update_info):
        self.llm.collective_rpc("update_weights", timeout=120, args=(update_info,))

    def finish_weight_update(self, weight_version=None):
        self.llm.collective_rpc("finish_weight_update", timeout=120)


def _run(directory: Path, task: str, slots: int, graph: bool):
    # Platform and Worker initialization must precede model/engine construction.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(directory / ("updated" if task == "cold" else "initial")),
        skip_tokenizer_init=True,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        enforce_eager=not graph,
        max_model_len=128,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        kv_cache_memory_bytes=256 * 2**20,
        enable_prefix_caching=False,
        weight_transfer_config={"backend": "hccl"},
        additional_config={"weight_nz_mode": 0},
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]}
        if graph
        else {},
        seed=0,
    )
    trainer = None
    try:
        pointers = llm.collective_rpc(_snapshot, args=(str(directory), task + "-before"))[0]
        if task == "update":
            from transformers import AutoModelForCausalLM
            from vllm.distributed.weight_transfer.base import ModuleSource
            from vllm.utils.network_utils import get_open_port

            from vllm_ascend.distributed.weight_transfer.hccl_engine import (
                HCCLTrainerInitInfo,
                HCCLTrainerWeightTransferEngine,
            )

            torch.npu.set_device(1)
            source = AutoModelForCausalLM.from_pretrained(directory / "updated", dtype=torch.bfloat16).to("npu:1")
            trainer = HCCLTrainerWeightTransferEngine.trainer_init(
                HCCLTrainerInitInfo(
                    rank=0,
                    master_address="127.0.0.1",
                    master_port=get_open_port(),
                    world_size=2,
                    packed=True,
                    packed_buffer_size_bytes=8192,
                    packed_num_buffers=slots,
                    packed_transfer_mode="bounded",
                ),
                client=_WorkerClient(llm),
                source=ModuleSource(source),
            )
            first = next(source.parameters()).detach().reshape(-1)[:1]
            original = first.clone()
            for step in range(3):
                first.copy_(original)
                if step < 2:
                    first.add_(0.125 * (step + 1))
                trainer.send_weights()
        outputs = llm.generate(
            [{"prompt_token_ids": [1, 17, 23]}, {"prompt_token_ids": [1, 111, 37, 51]}],
            SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=5),
            use_tqdm=False,
        )
        signature = [
            (
                list(result.outputs[0].token_ids),
                [
                    {token: value.logprob for token, value in distribution.items()}
                    for distribution in result.outputs[0].logprobs
                ],
            )
            for result in outputs
        ]
        final_pointers = llm.collective_rpc(_snapshot, args=(str(directory), task))[0]
        final = torch.load(directory / f"{task}-parameters.pt", weights_only=True)
        assert pointers == final_pointers
        if task == "update":
            initial = torch.load(directory / "update-before-parameters.pt", weights_only=True)
            assert any(not torch.equal(initial[name], final[name]) for name in final)
        torch.save((final, signature), directory / f"{task}.pt")
    finally:
        if trainer is not None:
            trainer.shutdown()
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="Requires two NPUs")
@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_bounded_dense_reload(tmp_path, slots, graph):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    for step, name in enumerate(("initial", "updated")):
        torch.manual_seed(1701 + step)
        model = Qwen2ForCausalLM(config)
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                parameter.normal_(1.0 if "norm.weight" in key else 0.0, 0.02)
        model.to(torch.bfloat16).save_pretrained(tmp_path / name)
    env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_ALLOW_INSECURE_SERIALIZATION="1")
    for task in ("cold", "update"):
        subprocess.run(
            [sys.executable, __file__, str(tmp_path), task, str(slots), str(int(graph))],
            env=env,
            timeout=480,
            check=True,
        )
    expected, cold_signature = torch.load(tmp_path / "cold.pt", weights_only=True)
    actual, updated_signature = torch.load(tmp_path / "update.pt", weights_only=True)
    assert expected.keys() == actual.keys()
    for name, tensor in actual.items():
        reference = expected[name]
        assert tensor.shape == reference.shape and tensor.dtype == reference.dtype, name
        assert torch.equal(tensor.reshape(-1).view(torch.uint8), reference.reshape(-1).view(torch.uint8)), name
    assert updated_signature == cold_signature


if __name__ == "__main__":
    _run(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), bool(int(sys.argv[4])))
