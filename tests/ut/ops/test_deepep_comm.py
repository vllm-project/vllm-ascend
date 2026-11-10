# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
from vllm_ascend.ops.fused_moe import deepep_comm as module
from vllm_ascend.ops.fused_moe import moe_comm_method as registry
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchOutput


class TestDeepEPComm(unittest.TestCase):
    def test_padding_trimming_and_hash_ids(self):
        for count, global_count in [(0, 0), (0, 5), (1, 5), (3, 5), (5, 5)]:
            with self.subTest(count=count, global_count=global_count):
                prepare = module.PrepareAndFinalizeWithDeepEP.__new__(module.PrepareAndFinalizeWithDeepEP)
                x = torch.arange(count * 4).reshape(count, 4).to(torch.bfloat16)
                logits = torch.ones(count, 8)
                with patch.object(module, "_EXTRA_CTX", SimpleNamespace(max_tokens_across_dp=global_count)):
                    output = prepare.prepare(x, logits)
                self.assertEqual(output.hidden_states.shape, (max(1, global_count), 4))
                self.assertTrue(torch.equal(prepare.finalize(output.hidden_states), x))
                self.assertEqual(torch.count_nonzero(output.hidden_states[count:]).item(), 0)
                ids = torch.arange(count)
                padded_ids = prepare.pad_and_split_input_ids(ids)
                self.assertEqual(padded_ids.numel(), max(1, global_count))
                self.assertTrue(torch.equal(padded_ids[:count], ids))

    def test_empty_expert_rank_still_combines(self):
        x = torch.empty(0, 4, dtype=torch.bfloat16)
        dispatcher = Mock()
        dispatcher.token_dispatch.return_value = MoETokenDispatchOutput(x, torch.tensor([0, 0]), 1, object())
        dispatcher.token_combine.return_value = x
        comm = module.DeepEPCommImpl.__new__(module.DeepEPCommImpl)
        comm.token_dispatcher = dispatcher
        with patch.object(module, "build_token_dispatch_input"), patch.object(module, "apply_moe_mlp") as mlp:
            result = comm.fused_experts(object(), object())
        mlp.assert_not_called()
        dispatcher.token_combine.assert_called_once()
        self.assertEqual(result.routed_out.shape, (0, 4))

    def test_expert_computation_keeps_original_moe_config(self):
        x = torch.ones(3, 4, dtype=torch.bfloat16)
        dispatcher = Mock()
        dispatcher.token_dispatch.return_value = MoETokenDispatchOutput(x, torch.tensor([3, 0]), 1, object())
        comm = module.DeepEPCommImpl.__new__(module.DeepEPCommImpl)
        comm.token_dispatcher = dispatcher
        comm.moe_config = SimpleNamespace(swiglu_limit=10.0)
        fused, scheme = object(), object()
        with (
            patch.object(module, "build_token_dispatch_input"),
            patch.object(module, "build_mlp_compute_input") as build,
            patch.object(module, "apply_moe_mlp", return_value=(x * 2, None)) as mlp,
        ):
            comm.fused_experts(fused, scheme)
        self.assertIs(build.call_args.kwargs["moe_config"], comm.moe_config)
        mlp.assert_called_once_with(build.return_value, scheme)
        self.assertTrue(torch.equal(dispatcher.token_combine.call_args.args[0], x * 2))

    def test_combine_weights_exactly_once_and_waits(self):
        dispatcher = module.TokenDispatcherWithDeepEP.__new__(module.TokenDispatcherWithDeepEP)
        dispatcher.buffer, dispatcher.calls = Mock(), 1
        event = Mock()
        dispatcher.buffer.combine.return_value = (torch.ones(2, 4), None, event)
        x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.bfloat16)
        metadata = module.DeepEPCombineMetadata(object(), torch.tensor([0.25, 0.5]))
        dispatcher.token_combine(x, metadata)
        weighted = dispatcher.buffer.combine.call_args.args[0]
        self.assertTrue(torch.equal(weighted, x * metadata.weights[:, None]))
        self.assertTrue(dispatcher.buffer.combine.call_args.kwargs["async_with_compute_stream"])
        event.current_stream_wait.assert_called_once()
        with self.assertRaisesRegex(ValueError, "bias"):
            dispatcher.token_combine(x, metadata, bias=x)

    def test_dispatch_invalidates_padding_routes_without_mutating_router_output(self):
        for count in [0, 1, 5]:
            with self.subTest(count=count):
                dispatcher = module.TokenDispatcherWithDeepEP.__new__(module.TokenDispatcherWithDeepEP)
                dispatcher.buffer, dispatcher.calls, dispatcher._initialize_buffer = Mock(), 0, Mock()
                dispatcher.prepare_finalize = SimpleNamespace(num_tokens=count)
                dispatcher.num_experts = 8
                x = torch.ones(5, 4, dtype=torch.bfloat16)
                ids, weights = torch.zeros(5, 2, dtype=torch.int64), torch.ones(5, 2)
                handle, event = SimpleNamespace(num_recv_tokens_per_expert_list=[2, 0]), Mock()
                dispatcher.buffer.dispatch.return_value = (x[:2], None, weights[:2, 0], handle, event)
                inputs = SimpleNamespace(hidden_states=x, topk_ids=ids, topk_weights=weights)
                result = dispatcher.token_dispatch(inputs)
                kwargs = dispatcher.buffer.dispatch.call_args.kwargs
                self.assertTrue(torch.equal(kwargs["topk_idx"][:count], ids[:count]))
                self.assertTrue(torch.all(kwargs["topk_idx"][count:] == -1))
                self.assertTrue(torch.all(kwargs["topk_weights"][count:] == 0))
                self.assertTrue(torch.all(ids == 0))
                self.assertTrue(torch.all(weights == 1))
                self.assertEqual(result.group_list.tolist(), [2, 0])
                self.assertEqual(result.group_list_type, 1)
                event.current_stream_wait.assert_called_once()

    def test_same_shape_layers_reuse_buffer_owner(self):
        config = SimpleNamespace(num_experts=384, num_local_experts=48)
        with (
            patch.dict(registry._MoECommMethods, {}, clear=True),
            patch.object(registry, "get_ascend_config", return_value=SimpleNamespace(enable_deepep=True)),
            patch.object(module, "DeepEPCommImpl") as implementation,
        ):
            registry.setup_moe_comm_method(config)
            registry.setup_moe_comm_method(config)
            implementation.assert_called_once_with(config)

    def test_private_shmem_integrity_and_symbol_isolation(self):
        names = ("libshmem_utils.so", "aclshmem_bootstrap_config_store.so", "aclshmem_bootstrap_uid.so", "libshmem.so")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle = directory / ".libs"
            bundle.mkdir()
            for name in names:
                (bundle / name).write_bytes(name.encode())
            manifest = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in names}
            (directory / "_build_info.json").write_text(json.dumps({"shmem_runtime": manifest}))
            native = SimpleNamespace(_loaded_shmem_paths=lambda: [])
            package = SimpleNamespace(__file__=str(directory / "__init__.py"))
            with (
                patch.dict(sys.modules, {"deep_ep": package, "deep_ep._native": native}),
                patch.object(module.os, "RTLD_DEEPBIND", 8, create=True),
                patch.object(module.ctypes, "RTLD_GLOBAL", 256, create=True),
                patch.object(module.ctypes, "RTLD_LOCAL", 0, create=True),
                patch.object(module.ctypes, "CDLL") as load,
            ):
                self.assertEqual(len(module._load_private_shmem()), 4)
                self.assertEqual(load.call_count, 4)
                self.assertTrue(all(call.kwargs["mode"] & 8 for call in load.call_args_list))
                load.reset_mock()
                (bundle / names[0]).write_bytes(b"corrupt")
                with self.assertRaisesRegex(ImportError, "integrity"):
                    module._load_private_shmem()
                load.assert_not_called()
                native._loaded_shmem_paths = lambda: [Path("/external/libshmem.so")]
                with self.assertRaisesRegex(ImportError, "Conflicting"):
                    module._load_private_shmem()

    def test_rejects_unsupported_configurations(self):
        cases = [
            ({"tensor_parallel_size": 2}, {}, {}, "TP1/DP8/EP8"),
            ({"data_parallel_size": 4}, {}, {}, "TP1/DP8/EP8"),
            ({"enable_expert_parallel": False}, {}, {}, "TP1/DP8/EP8"),
            ({}, {"enforce_eager": False}, {}, "enforce-eager"),
            ({"enable_eplb": True}, {}, {}, "EPLB"),
            ({}, {}, {"enable_fused_mc2": 1}, "fused MC2"),
            ({}, {}, {"mix_placement": True}, "mixed expert placement"),
        ]
        for parallel, model, ascend, message in cases:
            with self.subTest(parallel=parallel, model=model, ascend=ascend):
                pc = dict(enable_expert_parallel=True, tensor_parallel_size=1, data_parallel_size=8, enable_eplb=False)
                pc.update(parallel)
                mc = dict(enforce_eager=True)
                mc.update(model)
                config = SimpleNamespace(parallel_config=SimpleNamespace(**pc), model_config=SimpleNamespace(**mc))
                with self.assertRaisesRegex(ValueError, message):
                    AscendConfig(
                        sparse_kv_offload_config=SparseKVOffloadConfig(), enable_deepep=True, **ascend
                    ).derive_and_validate(config)
