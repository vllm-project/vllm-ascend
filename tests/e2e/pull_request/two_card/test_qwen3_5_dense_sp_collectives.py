# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
import torch_npu
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
    tensor_model_parallel_all_reduce,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.qwen3_next import Qwen3NextMLP
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter
from vllm_ascend.patch.worker.patch_qwen3_5 import AscendQwen3_5DecoderLayer


@torch.inference_mode()
def _worker(rank: int, port: int, model_path: str | None, result_queue: mp.SimpleQueue) -> None:
    try:
        torch_npu.npu.set_device(rank)
        result = {"rank": rank, "decoder_forward": True}
        config = (
            EngineArgs(
                model=model_path,
                tensor_parallel_size=2,
                trust_remote_code=True,
                quantization="ascend",
                max_model_len=128,
                additional_config={"enable_flashcomm1": True},
            ).create_engine_config()
            if model_path is not None
            else VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))
        )
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=2,
                rank=rank,
                local_rank=rank,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                backend="hccl",
            )
            ensure_model_parallel_initialized(2, 1)

            # Match the dense decoder contract: each rank starts with one token,
            # gathers the full token sequence, produces a rank-local TP partial,
            # then reduce-scatters the summed output back to one token per rank.
            local = torch.tensor([[rank * 2.0 + 1.0, rank * 2.0 + 2.0]], device="npu")
            gathered = sp_all_gather(local)
            expected_gathered = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
            torch.testing.assert_close(gathered, expected_gathered)

            partial = gathered * (rank + 1)
            sharded = sp_reduce_scatter(partial)
            expected_shard = (expected_gathered * 3)[rank : rank + 1]
            torch.testing.assert_close(sharded, expected_shard)

            # Exercise the actual dense-SP decoder forward with lightweight
            # rank-local attention/MLP partials. Each TP projection contributes
            # (rank + 1) * x, so both reduce-scatters must sum a factor of 3.
            layer = SimpleNamespace(
                _use_dense_sequence_parallel=True,
                layer_type="full_attention",
                layer_scale=False,
                input_layernorm=lambda hidden: hidden,
                self_attn=lambda **kwargs: kwargs["hidden_states"] * (rank + 1),
                post_attention_layernorm=lambda hidden, residual: (hidden, residual),
                mlp=lambda hidden: hidden * (rank + 1),
            )
            output, residual = AscendQwen3_5DecoderLayer.forward(
                layer,
                hidden_states=local,
                residual=None,
                positions=torch.arange(2, device="npu"),
            )
            torch.testing.assert_close(output, local * 9)
            torch.testing.assert_close(residual, local)

            if model_path is not None:
                text_config = config.model_config.hf_text_config
                with set_default_torch_dtype(torch.bfloat16), torch.device("npu"):
                    mlp = Qwen3NextMLP(
                        hidden_size=text_config.hidden_size,
                        intermediate_size=text_config.intermediate_size,
                        hidden_act=text_config.hidden_act,
                        quant_config=None,
                        reduce_results=False,
                        prefix="model.layers.0.mlp",
                    )
                for parameter in mlp.parameters():
                    parameter.fill_(0.0001 * (rank + 1))

                local_hidden = torch.linspace(
                    -0.25,
                    0.25,
                    text_config.hidden_size,
                    dtype=torch.bfloat16,
                    device="npu",
                ).unsqueeze(0)
                local_hidden = local_hidden + rank * 0.5
                full_hidden = sp_all_gather(local_hidden)
                partial_output = mlp(full_hidden)
                sp_output = sp_reduce_scatter(partial_output)
                reduced_output = tensor_model_parallel_all_reduce(partial_output.clone())
                expected_output = reduced_output[rank : rank + 1]
                torch.testing.assert_close(sp_output, expected_output, rtol=0.01, atol=0.01)
                result.update(
                    hidden_size=text_config.hidden_size,
                    intermediate_size=text_config.intermediate_size,
                    mlp_max_delta=(sp_output - expected_output).abs().max().item(),
                )

                with set_default_torch_dtype(torch.bfloat16), torch.device("npu"):
                    decoder = AscendQwen3_5DecoderLayer(
                        vllm_config=config,
                        layer_type="full_attention",
                        prefix="model.language_model.model.layers.3",
                    )
                assert decoder._use_dense_sequence_parallel
                assert decoder.use_attn_reduce_scatter_for_moe
                assert not decoder.self_attn.o_proj.reduce_results
                assert not decoder.mlp.down_proj.reduce_results
                result.update(
                    decoder_class=type(decoder).__name__,
                    attention_reduce_results=decoder.self_attn.o_proj.reduce_results,
                    mlp_reduce_results=decoder.mlp.down_proj.reduce_results,
                )
            torch_npu.npu.synchronize()
        result_queue.put(result)
    except Exception:
        result_queue.put(traceback.format_exc())
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.skipif(os.getenv("ASCEND_RT_VISIBLE_DEVICES") is None, reason="requires two visible NPUs")
def test_qwen3_5_dense_sp_real_hccl_collectives() -> None:
    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    port = 29_501 + random.randint(0, 10_000)
    model_path = os.getenv("SP_TEST_MODEL")
    processes = [context.Process(target=_worker, args=(rank, port, model_path, result_queue)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        results = [result_queue.get() for _ in processes]
        for process in processes:
            process.join(timeout=30)
        assert all(process.exitcode == 0 for process in processes)
        errors = [result for result in results if isinstance(result, str)]
        assert not errors, "\n".join(errors)
        assert sorted(result["rank"] for result in results) == [0, 1]
        assert all(result["decoder_forward"] for result in results)
        if model_path is not None:
            assert all(result["decoder_class"] == "AscendQwen3_5DecoderLayer" for result in results)
            assert all(not result["attention_reduce_results"] for result in results)
            assert all(not result["mlp_reduce_results"] for result in results)
        print(sorted(results, key=lambda result: result["rank"]))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


if __name__ == "__main__":
    test_qwen3_5_dense_sp_real_hccl_collectives()
