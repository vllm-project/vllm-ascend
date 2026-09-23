# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of integration lifecycle methods, with NPU events mocked.

Run: python -m tests.ut.worker.test_flash_mla_integration_host_contract -v
This is not an NPU scatter, Triton, operator, or graph execution test.
"""

import unittest
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from types import SimpleNamespace
from typing import Protocol, runtime_checkable
from unittest.mock import patch

import numpy as np
import torch

from tests.ut.attention import test_flash_mla_host_contract as attention_contract
from tests.ut.spec_decode import test_flash_mla_dspark_host_contract as dspark_contract


class FakeNPU:
    def __init__(self):
        self.calls = []
        self.active = self.Stream()
        self.capturing = False

    def Stream(self):
        calls = self.calls
        return SimpleNamespace(wait_event=lambda event: calls.append(("wait", event)))

    def Event(self):
        event = SimpleNamespace()
        event.record = lambda stream: self.calls.append(("record", event))
        return event

    def current_stream(self):
        return self.active

    def is_current_stream_capturing(self):
        return self.capturing

    @contextmanager
    def stream(self, stream):
        original = self.active
        self.active = stream
        try:
            yield
        finally:
            self.active = original


class FlashMLAIntegrationHostContract(unittest.TestCase):
    def setUp(self):
        self.fixture = attention_contract.FlashMLAHostContract()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.npu = FakeNPU()
        npu_patch = patch.object(torch, "npu", self.npu, create=True)
        npu_patch.start()
        self.addCleanup(npu_patch.stop)
        self.executor_module = attention_contract.load_functions(
            "vllm_ascend/worker/device_metadata.py",
            {"DeviceMetadataStage", "DeviceMetadataTask", "DeviceMetadataTaskProvider", "DeviceMetadataExecutor"},
            torch=torch,
            dataclass=dataclass,
            IntEnum=IntEnum,
            Protocol=Protocol,
            runtime_checkable=runtime_checkable,
        )
        self.owner = ContextVar("test_flash_owner", default=None)
        self.bridge = attention_contract.load_functions(
            "vllm_ascend/worker/v2/attn_utils.py",
            {"device_metadata_context", "build_attn_metadata"},
            torch=torch,
            np=np,
            contextmanager=contextmanager,
            _device_metadata_executor=self.owner,
            DeviceMetadataTaskProvider=self.executor_module.DeviceMetadataTaskProvider,
            AscendCommonAttentionMetadata=SimpleNamespace,
            AscendDSAMetadataBuilder=type("DSA", (), {}),
            AscendSFAMetadataBuilder=type("SFA", (), {}),
            GDNAttentionMetadataBuilder=type("GDN", (), {}),
        )

    def builder(self):
        builder = self.fixture.builder()

        def enable():
            builder._device_metadata_enabled = True

        def take():
            tasks = builder._device_metadata_tasks
            builder._device_metadata_tasks = ()
            return tasks

        def build(common_prefix_len, common_attn_metadata, **kwargs):
            return SimpleNamespace(
                flash=self.fixture.api._build_flash_attention_metadata(builder, common_attn_metadata)
            )

        builder.enable_device_metadata = enable
        builder.take_device_metadata_tasks = take
        builder.build = build
        builder.build_for_cudagraph_capture = lambda common, **kwargs: build(0, common)
        return builder

    def build(self, builder, common, capture=False):
        return self.bridge.build_attn_metadata(
            attn_groups=[[SimpleNamespace(get_metadata_builder=lambda i: builder, layer_names=["mla"])]],
            num_reqs=common.num_reqs,
            num_tokens=common.num_input_tokens,
            query_start_loc_gpu=common.query_start_loc,
            query_start_loc_cpu=common.query_start_loc.clone(),
            max_query_len=common.max_query_len,
            seq_lens=common.seq_lens,
            max_seq_len=128000,
            block_tables=[common.block_table_tensor],
            slot_mappings=[common.slot_mapping],
            kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
            positions=common.positions,
            causal=common.causal,
            for_cudagraph_capture=capture,
        )["mla"].flash

    def test_capture_then_replay_refreshes_buffers_after_consumer_fence(self):
        executor = self.executor_module.DeviceMetadataExecutor()
        builder, common = self.builder(), self.fixture.common(causal=False)
        with self.bridge.device_metadata_context(executor):
            captured = self.build(builder, common, capture=True)
            self.assertTrue(executor.submission_in_flight)
            ready = executor._stage_ready[(2, id(captured.schedule))]
            self.assertTrue(any(kind == "wait" and event is ready for kind, event in self.npu.calls))
            _, cache = self.fixture.cache()
            self.fixture.api.run_flash_mla(captured.query, cache, captured, 1)
        self.assertFalse(executor.submission_in_flight)
        self.assertIs(self.npu.calls[-1][1], executor._buffer_reusable)
        common.seq_lens.fill_(127)
        common.block_table_tensor.add_(1)
        with self.bridge.device_metadata_context(executor):
            replayed = self.build(builder, common)
            self.assertIs(captured, replayed)
            self.assertEqual(replayed.cache_lens[0], 127)
            self.assertTrue(
                any(kind == "wait" and event is executor._buffer_reusable for kind, event in self.npu.calls)
            )
            before = len(self.npu.calls)
            executor.wait(2, id(captured.schedule))
            self.assertEqual(len(self.npu.calls), before)  # no wait captured twice
        self.assertEqual([call[0] for call in self.fixture.calls], ["metadata", "metadata", "main", "metadata"])
        self.assertIsNone(self.owner.get())

    def test_target_draft_executors_and_frontiers_are_independent(self):
        target, draft = (self.executor_module.DeviceMetadataExecutor() for _ in range(2))
        with self.bridge.device_metadata_context(target):
            target_flash = self.build(self.builder(), self.fixture.common())
            with self.bridge.device_metadata_context(draft):
                draft_flash = self.build(self.builder(), self.fixture.common(causal=False))
                self.assertIs(self.owner.get(), draft)
                self.assertNotEqual(target_flash.schedule.data_ptr(), draft_flash.schedule.data_ptr())
            self.assertTrue(target.submission_in_flight)
            self.assertFalse(draft.submission_in_flight)
            self.assertIs(self.owner.get(), target)
        self.assertFalse(target.submission_in_flight)

    def test_schedule_update_is_rejected_inside_capture(self):
        executor = self.executor_module.DeviceMetadataExecutor()
        self.npu.capturing = True
        with self.bridge.device_metadata_context(executor), self.assertRaises(AssertionError):
            self.build(self.builder(), self.fixture.common())
        self.assertFalse(executor.submission_in_flight)
        self.assertEqual(self.fixture.generations, 0)

    def test_executor_rejects_overwrite_before_release(self):
        executor = self.executor_module.DeviceMetadataExecutor()
        tasks = [self.executor_module.DeviceMetadataTask(2, lambda: None, 1)]
        executor.submit(tasks)
        with self.assertRaisesRegex(RuntimeError, "not been released"):
            executor.submit(tasks)
        executor.wait(2, 1)
        executor.release()
        executor.submit(tasks)
        executor.wait(2, 1)
        executor.release()

    def test_layer_keeps_v_up_gate_o_proj_and_masks_dead_rows(self):
        for rope in (False, True):
            for fused in (False, True):
                with self.subTest(rope=rope, fused=fused):
                    self.check_layer_output(rope, fused)

    def check_layer_output(self, rope, fused):
        # Execute the real projection/control flow; operator math, RoPE,
        # scatter, and Triton masks remain CPU substitutes in this test.
        class LinearMethod:
            pass

        class OutputProjection:
            input_is_parallel = True
            bias = None
            custom_op = None
            quant_method = LinearMethod()
            weight = torch.ones(4, 16, dtype=torch.bfloat16)

            def __call__(self, value, **kwargs):
                return (value @ self.weight.t(),)

        common = self.fixture.common(query=2, padding=2)
        common.slot_mapping = torch.tensor([127, 128])
        flash = self.fixture.api._build_flash_attention_metadata(self.fixture.builder(), common)
        raw, cache = self.fixture.cache()
        expected = raw.clone()
        view = torch.as_strided(expected, cache.shape, cache.stride(), cache.storage_offset())
        kv = torch.arange(4 * 576).view(4, 576).to(torch.bfloat16)
        for row, slot in enumerate((127, 128)):
            view[slot // 128, slot % 128, 0, :512] = kv[row, :512]
            view[slot // 128, slot % 128, 0, 512:] = kv[row, 512:] + int(rope)
        calls = []

        def scatter(**kwargs):
            self.assertEqual(kwargs["key_cache"].stride(), cache.stride())
            for row, slot in enumerate(kwargs["slot_mapping"].tolist()):
                if slot >= 0:
                    kwargs["key_cache"][slot // 128, slot % 128] = kwargs["key"][row]
                    kwargs["value_cache"][slot // 128, slot % 128] = kwargs["value"][row]

        def main(query, consumed_cache, metadata, scale):
            calls.append("main")
            self.assertIs(consumed_cache, cache)
            self.assertIs(metadata, flash)
            self.assertEqual(query.shape, (4, 8, 576))
            self.assertTrue((query[..., 512:] == 1 + int(rope)).all())
            latent = torch.ones(8, 4, 512, dtype=torch.bfloat16)
            latent[:, 2:] = float("nan")
            return latent, torch.empty(0)

        def gate(projected, logits, live):
            calls.append("fused-gate")
            projected.copy_(torch.where(live[:, None], projected * torch.sigmoid(logits), 0))

        def output(result, live, out):
            calls.append("output-mask")
            return out.copy_(torch.where(live[:, None], result, 0))

        scope = dspark_contract.load_selected(
            "vllm_ascend/attention/mla_v1.py",
            classes={"AscendMLAImpl": {"_forward_flash", "_q_proj_and_k_up_proj", "_v_up_proj"}},
            MLAAttentionImpl=object,
            torch=torch,
            torch_npu=SimpleNamespace(
                npu_scatter_pa_kv_cache=scatter,
                npu_transpose_batchmatmul=lambda x, w, **kw: torch.bmm(x, w).transpose(0, 1),
            ),
            DeviceMetadataStage=SimpleNamespace(ATTENTION=2),
            wait_for_device_metadata=lambda *a: calls.append("wait"),
            validate_flash_cache=self.fixture.api.validate_flash_cache,
            notify_kv_cache_written=lambda *a: calls.append("write"),
            run_flash_mla=main,
            get_cos_and_sin_mla=lambda *a, **kw: (None, None),
            flash_attention_gate=gate,
            flash_attention_output=output,
            UnquantizedLinearMethod=LinearMethod,
        )
        impl = scope.AscendMLAImpl()
        impl.num_heads, impl.kv_lora_rank, impl.v_head_dim = 8, 512, 2
        impl.qk_head_dim, impl.qk_nope_head_dim, impl.qk_rope_head_dim = 66, 2, 64
        impl.fused_qkv_a_proj, impl.layerwise_kv_cache_hook = None, None
        impl.kv_a_proj_with_mqa = lambda x: (kv,)
        impl.q_proj = lambda x: (torch.ones(4, 8 * 66, dtype=torch.bfloat16),)
        impl.W_UK_T = torch.ones(8, 2, 512, dtype=torch.bfloat16)
        impl.W_UV = torch.ones(8, 512, 2, dtype=torch.bfloat16)
        impl.kv_a_layernorm = lambda x: x
        impl.use_mla_rope, impl.use_output_gate, impl.scale = rope, True, 1
        impl.rope_single = lambda x, cos, sin: x + 1
        impl.g_proj = lambda x: (torch.zeros(4, 16, dtype=torch.bfloat16),)
        impl.o_proj = OutputProjection()
        impl.o_proj.reduce_results = not fused
        out = torch.empty(4, 4, dtype=torch.bfloat16)
        result = impl._forward_flash(
            "mla", torch.ones(4, 4, dtype=torch.bfloat16), cache, SimpleNamespace(flash=flash), out
        )
        self.assertIs(result, out)
        torch.testing.assert_close(out[:2], torch.full((2, 4), 4096, dtype=torch.bfloat16))
        self.assertTrue((out[2:] == 0).all())
        torch.testing.assert_close(raw, expected, rtol=0, atol=0)
        self.assertEqual(calls, ["wait", "write", "main", "fused-gate" if fused else "output-mask"])

    def test_metadata_refinement_leaves_non_mla_groups_unchanged(self):
        class MLA:
            pass

        class GQA:
            pass

        class Group:
            def __init__(self, backend, names, spec, group_id):
                self.backend, self.layer_names = backend, names
                self.kv_cache_spec, self.kv_cache_group_id = spec, group_id
                self.metadata_builders = [object()]

            def create_metadata_builders(self, **kwargs):
                self.kwargs = kwargs

        mla = Group(MLA, ["mla.0", "mla.1"], object(), 0)
        gqa = Group(GQA, ["gqa.1", "gqa.0"], object(), 1)
        groups, support, sizes = [[mla], [gqa]], object(), [128, 128]
        layers = {
            "mla.0": SimpleNamespace(impl=SimpleNamespace(use_mla_rope=True, scale=1)),
            "mla.1": SimpleNamespace(impl=SimpleNamespace(use_mla_rope=False, scale=1)),
            "gqa.0": SimpleNamespace(impl=SimpleNamespace(scale=1)),
            "gqa.1": SimpleNamespace(impl=SimpleNamespace(scale=2)),
        }
        module = attention_contract.load_functions(
            "vllm_ascend/worker/v2/attn_utils.py",
            {"init_attn_backend"},
            _upstream_init_attn_backend=lambda *a, **kw: (groups, support, sizes),
            ascend_envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            get_layers_from_vllm_config=lambda *a: layers,
            AttentionLayerBase=object,
            AscendMLABackend=MLA,
            AttentionGroup=Group,
        )
        result, actual_support, actual_sizes = module.init_attn_backend(object(), object(), "cpu")
        self.assertIs(actual_support, support)
        self.assertIs(actual_sizes, sizes)
        self.assertEqual([g.layer_names for g in result[0]], [["mla.0"], ["mla.1"]])
        self.assertIs(result[1][0], gqa)
        self.assertEqual(gqa.layer_names, ["gqa.1", "gqa.0"])
        self.assertTrue(all(g.kv_cache_spec is mla.kv_cache_spec for g in result[0]))

    def test_cow_retains_strided_views_and_copies_all_kernel_pages(self):
        copy = attention_contract.load_functions(
            "vllm_ascend/worker/utils.py",
            {"copy_kv_cache_blocks_inplace"},
            torch=torch,
            np=np,
            async_tensor_h2d=lambda array, device: torch.from_numpy(array).to(device),
        ).copy_kv_cache_blocks_inplace
        # Two allocator blocks, each split into three 128-token kernel pages.
        slot_elements, offset = 81408, 38
        raw = torch.full((6 * slot_elements + offset,), -9, dtype=torch.bfloat16)
        cache = torch.as_strided(raw, (6, 128, 1, 576), (slot_elements, 576, 576, 1), offset)
        before = raw.clone()
        cache[3:].copy_(torch.arange(3, dtype=torch.bfloat16).view(3, 1, 1, 1) + 1)
        expected = raw.clone()
        expected_cache = torch.as_strided(expected, cache.shape, cache.stride(), offset)
        expected_cache[:3].copy_(cache[3:])
        pointer, strides = cache.data_ptr(), cache.stride()
        copy([cache, cache], 2, [SimpleNamespace(src_block_id=1, dst_block_id=0)])
        self.assertEqual((cache.data_ptr(), cache.stride(), cache.storage_offset()), (pointer, strides, offset))
        torch.testing.assert_close(raw, expected, rtol=0, atol=0)
        torch.testing.assert_close(raw[:offset], before[:offset], rtol=0, atol=0)

    def test_zeroer_metadata_keeps_original_fused_page_span(self):
        class Spec:
            block_size = 384

        module = attention_contract.load_functions(
            "vllm_ascend/worker/utils.py",
            {"_component_views_share_slot", "AscendKVBlockZeroer"},
            torch=torch,
            KVBlockZeroer=object,
            FullAttentionSpec=Spec,
            MLAAttentionSpec=Spec,
            iprod=product,
            largest_power_of_2_divisor=lambda n: n & -n,
        )
        raw = torch.empty(6 * 81408, dtype=torch.bfloat16)
        cache = torch.as_strided(raw, (6, 128, 1, 576), (81408, 576, 576, 1))
        zeroer = module.AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
        zeroer.init_meta(
            [SimpleNamespace(kv_cache_spec=Spec(), kv_cache_group_id=0, layer_names=["mla"])],
            [[128]],
            "auto",
            set(),
            {"mla": SimpleNamespace(kv_cache=cache)},
        )
        addresses, page_elements, _, count = zeroer._meta
        self.assertEqual(addresses.tolist(), [cache.data_ptr()])
        self.assertEqual(page_elements * 4, 488448)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
