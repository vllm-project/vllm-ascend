# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU behavior tests for the production snapshot and admission methods."""

import ast
import importlib.util
import sys
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = ROOT / "vllm_ascend/worker/v2/model_runner.py"
HELPER_PATH = RUNNER_PATH.with_name("mm_decode_queue.py")
SPEC = importlib.util.spec_from_file_location("_test_mm_decode_queue_runner_helper", HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
# Load only the CPU proof helper; do not import the NPU runner dependencies.
with patch.dict(sys.modules, {SPEC.name: HELPER}):
    SPEC.loader.exec_module(HELPER)
PD_SPEC = importlib.util.spec_from_file_location("_test_pd_runner_helper", RUNNER_PATH.with_name("pd_decode_queue.py"))
PD_HELPER = importlib.util.module_from_spec(PD_SPEC)
with patch.dict(sys.modules, {PD_SPEC.name: PD_HELPER}):
    PD_SPEC.loader.exec_module(PD_HELPER)


class BaseRunner:
    def finish_requests(self, output):
        self.lifecycle_log.append("base_finish")


METHODS = {
    "finish_requests",
    "_qwen_mrv2_mm_decode_request_keys",
    "_qwen_mrv2_mm_decode_context",
    "_qwen_mrv2_can_queue_decode",
    "_qwen_mrv2_has_local_uniform_verification",
    "_qwen_mrv2_mm_batch_is_stable_decode",
    "_qwen_mrv2_mm_decode_is_certified",
    "_qwen_mrv2_certify_mm_decode",
    "_qwen_mrv2_certify_pd_decode",
    "_update_seq_lens_cpu",
}
# Execute the production methods unchanged with explicit CPU stand-ins. This
# checks event ordering and admission behavior, not NPU execution or overlap.
runner_ast = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
runner_class = next(
    node for node in runner_ast.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
)
assert {node.name for node in runner_class.body if isinstance(node, ast.FunctionDef)} >= METHODS

test_class = ast.ClassDef(
    name="RunnerMethods",
    bases=[ast.Name(id="BaseRunner", ctx=ast.Load())],
    keywords=[],
    body=[node for node in runner_class.body if isinstance(node, ast.FunctionDef) and node.name in METHODS],
    decorator_list=[],
)
tree = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
namespace = {
    "BaseRunner": BaseRunner,
    "SchedulerOutput": object,
    "BatchReqState": object,
    "BatchExecutionDescriptor": object,
    "np": np,
    "torch": NS(npu=NS(current_stream=lambda: "main")),
    "make_mm_decode_request_key": HELPER.make_mm_decode_request_key,
    "ObjectIdentity": HELPER.ObjectIdentity,
    "CUDAGraphMode": NS(FULL_DECODE_ONLY=object(), FULL=object()),
    "has_kv_transfer_group": lambda: False,
    "PDDecodeQueueKVConnector": type("PDDecodeQueueKVConnector", (), {}),
}
exec(compile(tree, str(RUNNER_PATH), "exec"), namespace)
Runner = namespace["RunnerMethods"]


def scheduler(**changes):
    output = NS(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=NS(req_ids=["image"], resumed_req_ids=[]),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_encoder_inputs={},
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={"image": 4},
    )
    vars(output).update(changes)
    return output


def make_runner(exact=100, upper=104):
    runner = Runner()
    runner._qwen_mrv2_mm_decode_proof_enabled = True
    runner._qwen_mrv2_mm_decode_epoch = 0
    runner._qwen_mrv2_mm_decode_state = HELPER.MMDecodeCertificateState()
    runner._qwen_mrv2_mm_decode_stats = {"invalidations": 0, "exact_certified_batches": 0}
    runner._qwen_mrv2_mm_snapshot_stream = "main"
    runner._qwen_mrv2_early_queue_stats = {"host_length_wait_skipped": 0, "length_device_wait_enqueued": 0}
    runner._qwen_mrv2_mm_decode_context = lambda: (runner._qwen_mrv2_mm_decode_epoch,)
    runner.speculator = object()
    runner.sync_spec_pp_cpu_counts = False
    runner._can_skip_spec_cpu_length_sync = lambda: False
    runner.req_states = NS(req_id_to_index={"image": 0}, num_computed_tokens_cpu=np.array([upper]))
    runner.encoder_cache = NS(
        mm_features={
            "image": [
                NS(
                    data=object(),
                    modality="image",
                    identifier="image-hash",
                    mm_position=NS(offset=20, length=30, is_embed=None),
                )
            ]
        }
    )
    runner.input_buffers = NS(seq_lens_cpu=np.zeros(1, dtype=np.int32))
    runner.lifecycle_log = []
    runner.synchronized = False

    class PinnedSnapshot:
        def __getitem__(self, index):
            assert runner.synchronized, "exact data read before completion"
            return exact

        def numpy(self):
            assert runner.synchronized, "certificate read before completion"
            return np.array([exact])

    def sync():
        runner.synchronized = True

    runner.num_computed_tokens_cpu = PinnedSnapshot()
    runner.num_computed_tokens_event = NS(synchronize=sync)
    return runner


def batch():
    return NS(
        req_ids=["image"],
        idx_mapping_np=np.array([0]),
        prefill_len_np=np.array([100]),
        has_prefill=False,
        is_prefilling_np=np.array([False]),
        prefill_runs_as_decode_np=None,
    )


class TestRunnerSnapshotAdmission(unittest.TestCase):
    def record(self, runner, current_batch):
        keys = runner._qwen_mrv2_mm_decode_request_keys(
            current_batch.req_ids, current_batch.idx_mapping_np, current_batch.prefill_len_np
        )
        runner._qwen_mrv2_mm_decode_state.record_snapshot(keys)

    def test_optimistic_decode_flags_cannot_certify_incomplete_prompt(self):
        runner, current_batch, output = make_runner(exact=99), batch(), scheduler()
        self.record(runner, current_batch)
        runner._update_seq_lens_cpu(output, current_batch.req_ids)
        runner._qwen_mrv2_certify_mm_decode(output, current_batch)
        self.assertFalse(runner._qwen_mrv2_mm_decode_is_certified(output, current_batch))

    def test_synchronized_exact_boundary_enables_following_decode(self):
        runner, current_batch, output = make_runner(), batch(), scheduler()
        self.record(runner, current_batch)
        self.assertFalse(runner._qwen_mrv2_mm_decode_is_certified(output, current_batch))
        runner._update_seq_lens_cpu(output, current_batch.req_ids)
        runner._qwen_mrv2_certify_mm_decode(output, current_batch)
        self.assertTrue(runner._qwen_mrv2_mm_decode_is_certified(output, current_batch))
        self.assertEqual(runner.input_buffers.seq_lens_cpu.tolist(), [104])

    def test_fastpath_queues_device_dependency_without_reading_pinned_copy(self):
        runner, current_batch, output = make_runner(), batch(), scheduler()
        waits = []
        namespace["torch"].npu.current_stream = lambda: NS(wait_event=waits.append)
        try:
            runner._update_seq_lens_cpu(output, current_batch.req_ids, allow_cpu_upper_bound=True)
        finally:
            namespace["torch"].npu.current_stream = lambda: "main"
        self.assertFalse(runner.synchronized)
        self.assertEqual(waits, [runner.num_computed_tokens_event])
        self.assertEqual(runner.input_buffers.seq_lens_cpu.tolist(), [108])

    def test_slot_reuse_cannot_consume_previous_request_snapshot(self):
        runner, current_batch, output = make_runner(), batch(), scheduler()
        self.record(runner, current_batch)
        runner.finish_requests(scheduler(scheduled_new_reqs=[object()]))
        runner._update_seq_lens_cpu(output, current_batch.req_ids)
        runner._qwen_mrv2_certify_mm_decode(output, current_batch)
        self.assertFalse(runner._qwen_mrv2_mm_decode_is_certified(output, current_batch))
        self.assertEqual(runner.lifecycle_log, ["base_finish"])

    def test_all_scheduler_lifecycle_changes_revoke_proof(self):
        for changes in (
            dict(scheduled_new_reqs=[object()]),
            dict(finished_req_ids={"image"}),
            dict(preempted_req_ids={"image"}),
            dict(scheduled_encoder_inputs={"image": [0]}),
            dict(free_encoder_mm_hashes=["image-hash"]),
            dict(scheduled_cached_reqs=NS(req_ids=["image"], resumed_req_ids=["image"])),
        ):
            with self.subTest(changes=changes):
                runner, current_batch, output = make_runner(), batch(), scheduler()
                self.record(runner, current_batch)
                runner._update_seq_lens_cpu(output, current_batch.req_ids)
                runner._qwen_mrv2_certify_mm_decode(output, current_batch)
                self.assertTrue(runner._qwen_mrv2_mm_decode_is_certified(output, current_batch))
                runner.finish_requests(scheduler(**changes))
                self.assertFalse(runner._qwen_mrv2_mm_decode_is_certified(output, current_batch))

    def test_changed_media_or_producer_stream_requires_fresh_exact_proof(self):
        for mutation in ("media", "stream"):
            runner, current_batch, output = make_runner(), batch(), scheduler()
            self.record(runner, current_batch)
            if mutation == "media":
                runner.encoder_cache.mm_features["image"][0].mm_position.offset += 1
            else:
                runner._qwen_mrv2_mm_snapshot_stream = "unrelated-stream"
            runner._update_seq_lens_cpu(output, current_batch.req_ids)
            runner._qwen_mrv2_certify_mm_decode(output, current_batch)
            self.assertFalse(runner._qwen_mrv2_mm_decode_is_certified(output, current_batch))


class TestMMConsumerPolicy(unittest.TestCase):
    def make_runner(self):
        class QwenState:
            supports_mm_inputs = True
            mm_pruner = None
            prompt_embeds_state = None
            rope_state = object()

            def prepare_inputs_embeds(self):
                pass

            def gather_mm_embeddings(self):
                pass

        class Encoder:
            is_realtime = False

            def gather_mm_embeddings(self):
                pass

        module_name = "vllm_ascend.models.qwen4_exp.model_state"
        old_module = sys.modules.get(module_name)
        sys.modules[module_name] = NS(AscendQwen4ExpModelState=QwenState)
        self.addCleanup(self.restore_module, module_name, old_module)
        runner = Runner()
        runner._qwen_mrv2_mm_decode_proof_enabled = True
        runner._qwen_mrv2_mm_decode_epoch = 3
        runner.req_states = object()
        runner.supports_mm_inputs = True
        runner.encoder_cache = object()
        runner.model_state = QwenState()
        runner.model_state.encoder_cache = runner.encoder_cache
        runner.model_state.encoder_runner = Encoder()
        runner.model_state.encoder_runner.encoder_cache = runner.encoder_cache
        runner.model = NS(deepstack_input_embeds_num_tokens=0)
        runner.model_config = NS(
            enable_prompt_embeds=False, is_mm_prefix_lm=False, is_encoder_decoder=False, hf_text_config=object()
        )
        return runner

    def restore_module(self, name, old_module):
        if old_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old_module

    def test_stable_model_and_rope_identity_are_part_of_snapshot_key(self):
        runner = self.make_runner()
        before = runner._qwen_mrv2_mm_decode_context()
        self.assertIsNotNone(before)
        runner.model_state.rope_state = object()
        self.assertNotEqual(before, runner._qwen_mrv2_mm_decode_context())

    def test_unaudited_mm_consumers_and_active_deepstack_force_fallback(self):
        for target, attr, value in (
            ("encoder", "is_realtime", True),
            ("encoder", "encoder_cache", object()),
            ("state", "mm_pruner", object()),
            ("state", "prompt_embeds_state", object()),
            ("model", "deepstack_input_embeds_num_tokens", 8),
            ("config", "enable_prompt_embeds", True),
            ("config", "is_mm_prefix_lm", True),
            ("config", "is_encoder_decoder", True),
        ):
            with self.subTest(target=target, attr=attr):
                runner = self.make_runner()
                obj = {
                    "encoder": runner.model_state.encoder_runner,
                    "state": runner.model_state,
                    "model": runner.model,
                    "config": runner.model_config,
                }[target]
                setattr(obj, attr, value)
                self.assertIsNone(runner._qwen_mrv2_mm_decode_context())

    def queue_environment_runner(self):
        runner = self.make_runner()

        class KVConnector:
            pass

        class ECConnector:
            pass

        class AscendMTPSpeculator:
            pass

        namespace["KVConnector"] = KVConnector
        for name, exports in (
            ("vllm.v1.worker.gpu.ec_connector", NS(ECConnector=ECConnector)),
            ("vllm_ascend.worker.v2.spec_decode.mtp.speculator", NS(AscendMTPSpeculator=AscendMTPSpeculator)),
        ):
            self.addCleanup(self.restore_module, name, sys.modules.get(name))
            sys.modules[name] = exports
        runner.speculator = AscendMTPSpeculator()
        runner.speculative_config = NS(method="mtp", uses_dynamic_speculative_decoding=lambda: False)
        runner.num_speculative_steps = 3
        runner.compilation_config = NS(cudagraph_mode=namespace["CUDAGraphMode"].FULL_DECODE_ONLY)
        runner.vllm_config = NS(
            use_cumem_cudagraph_pool=False,
            offload_config=NS(
                uva=NS(cpu_offload_gb=0, cpu_offload_params=[]), prefetch=NS(offload_group_size=0, offload_params=[])
            ),
        )
        runner.uses_inputs_embeds = True
        runner.is_first_pp_rank = True
        runner.parallel_config = NS(
            tensor_parallel_size=8,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            use_sequence_parallel_moe=False,
            enable_dbo=False,
        )
        runner.ascend_config = NS(enable_sp_by_pass=False)
        runner.lora_config = None
        runner.cache_config = NS(enable_prefix_caching=False, mamba_cache_mode="none", kv_sharing_fast_prefill=False)
        runner.kv_connector = KVConnector()
        runner.ec_connector = ECConnector()
        runner.aux_output_connector = None
        runner.adaptive_verification = None
        runner.sparse_kv_offload_manager = None
        return runner

    def queue_runner(self, media, *, draft_tokens=3):
        runner = self.queue_environment_runner()
        runner.num_speculative_steps = draft_tokens
        runner._qwen_mrv2_early_queue_enabled = True
        runner._qwen_mrv2_mm_decode_queue_enabled = False
        runner._qwen_mrv2_snapshot_generation = 1
        runner._qwen_mrv2_snapshot_source_stream = "main"
        runner._qwen_mrv2_early_queue_stats = {"fallback_steps": 0, "eligible_steps": 0}
        runner._qwen_mrv2_mm_decode_stats = {"queue_eligible_steps": 0}
        runner.sync_spec_pp_cpu_counts = False
        runner._can_skip_spec_cpu_length_sync = lambda: False
        runner.scheduler_config = NS(async_scheduling=True)
        runner.max_model_len = 141312
        runner.req_states = NS(req_id_to_index={"image": 0}, num_computed_tokens_np=np.array([104]))
        runner.encoder_cache = NS(mm_features={"image": media})
        runner.model_state.encoder_cache = runner.encoder_cache
        runner.model_state.encoder_runner.encoder_cache = runner.encoder_cache
        runner._qwen_mrv2_mm_decode_state = HELPER.MMDecodeCertificateState()

        class Builder:
            pass

        class QSA:
            @classmethod
            def get_impl_cls(cls):
                return NS

        group = NS(backend=QSA, get_metadata_builder=lambda _: Builder())
        runner.attn_groups = [[group]]
        runner.speculator.attn_groups = [[group]]
        runner.speculator.attn_architecture = "QSA"
        runner.speculator.attn_backend = QSA
        runner.speculator.draft_model_config = NS(max_model_len=141312)
        runner._qwen_mrv2_allowed_backend_builders = {QSA: (Builder,)}
        runner._qwen_mrv2_qsa_backend = QSA
        runner._qwen_mrv2_qsa_impl = NS
        runner._qwen_mrv2_backend_proof_key = None
        runner._qwen_mrv2_backend_proof = False
        current_batch = batch()
        current_batch.num_scheduled_tokens = np.array([draft_tokens + 1])
        current_batch.num_tokens = draft_tokens + 1
        current_batch.num_draft_tokens_np = np.array([draft_tokens])
        output = scheduler(
            num_scheduled_tokens={"image": draft_tokens + 1},
            scheduled_spec_decode_tokens={"image": list(range(draft_tokens))},
            has_structured_output_requests=False,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
            ec_manager_metadata=None,
            kv_cache_block_copies=[],
        )
        desc = NS(
            cg_mode=namespace["CUDAGraphMode"].FULL,
            uniform_token_count=draft_tokens + 1,
            num_tokens=draft_tokens + 1,
            num_reqs=1,
        )
        return runner, current_batch, output, desc

    def test_existing_text_queue_flag_behavior_is_unchanged(self):
        runner, current_batch, output, desc = self.queue_runner([])
        runner._qwen_mrv2_mm_decode_proof_enabled = False
        self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
        self.assertEqual(runner._qwen_mrv2_mm_decode_stats["queue_eligible_steps"], 0)
        runner._qwen_mrv2_early_queue_enabled = False
        self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))

    def pd_queue_runner(self):
        runner, current_batch, output, desc = self.queue_runner([])
        connector = namespace["PDDecodeQueueKVConnector"]()
        connector.transfer_group = object()
        connector.decode_queue_state = PD_HELPER.PDDecodeQueueState()
        runner.kv_connector = connector
        output.kv_connector_metadata = NS(requests={}, requests_to_send={}, reqs_in_batch=set())
        output.has_sync_kv_loads = False
        connector.decode_queue_state.observe_step(output, metadata_known=True)
        connector.decode_queue_state.record_snapshot({"image": 0}, runner._qwen_mrv2_snapshot_generation)
        runner.num_computed_tokens_cpu = NS(numpy=lambda: np.array([100]))
        self.enterContext(
            patch.dict(
                namespace,
                {
                    "has_kv_transfer_group": lambda: True,
                    "get_kv_transfer_group": lambda: connector.transfer_group,
                },
            )
        )
        return runner, current_batch, output, desc

    def test_pd_queue_requires_exact_stable_prompt_certificate(self):
        runner, current_batch, output, desc = self.pd_queue_runner()
        self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
        runner._qwen_mrv2_certify_pd_decode(output, current_batch)
        self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
        self.assertEqual(runner.kv_connector.decode_queue_state.stats["exact_certified_batches"], 1)

    def test_pd_preserves_parallel_prefix_and_ple_generalization(self):
        runner, current_batch, output, desc = self.pd_queue_runner()
        runner._qwen_mrv2_certify_pd_decode(output, current_batch)
        for tp, dp, pp, sp, prefix, offload in product(
            (1, 2, 4), (1, 2), (1, 2), (False, True), (False, True), (False, True)
        ):
            runner.parallel_config.tensor_parallel_size = tp
            runner.parallel_config.data_parallel_size = dp
            runner.parallel_config.pipeline_parallel_size = pp
            runner.parallel_config.use_sequence_parallel_moe = sp
            runner.ascend_config.enable_sp_by_pass = sp
            runner.cache_config.enable_prefix_caching = prefix
            runner.vllm_config.engram_config = NS(cpu_offload=offload)
            with self.subTest(tp=tp, dp=dp, pp=pp, sp=sp, prefix=prefix, offload=offload):
                self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))

    def test_pd_sync_load_copy_or_connector_change_keeps_baseline(self):
        for mutation in ("sync_load", "cache_copy", "group_change"):
            with self.subTest(mutation=mutation):
                runner, current_batch, output, desc = self.pd_queue_runner()
                runner._qwen_mrv2_certify_pd_decode(output, current_batch)
                self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                if mutation == "sync_load":
                    output.has_sync_kv_loads = True
                elif mutation == "cache_copy":
                    output.kv_cache_block_copies = [(1, 2)]
                else:
                    namespace["get_kv_transfer_group"] = object
                self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))

    def test_optional_a5_state_builder_requires_exact_audited_identity(self):
        class StateBuilder:
            pass

        class AscendQSAStateMetadataBuilder(StateBuilder):
            pass

        class UnknownChild(AscendQSAStateMetadataBuilder):
            pass

        class CPUReadingOverride(AscendQSAStateMetadataBuilder):
            def build(self, state):
                return state.num_computed_tokens_cpu.numpy()

        class StateBackend:
            pass

        for export, actual_builder, admitted in (
            (AscendQSAStateMetadataBuilder, AscendQSAStateMetadataBuilder, True),
            (AscendQSAStateMetadataBuilder, UnknownChild, False),
            (AscendQSAStateMetadataBuilder, CPUReadingOverride, False),
            (None, AscendQSAStateMetadataBuilder, False),
            (42, AscendQSAStateMetadataBuilder, False),
            (object, AscendQSAStateMetadataBuilder, False),
            (None, StateBuilder, True),
        ):
            with self.subTest(export=export, actual_builder=actual_builder):
                runner, current_batch, output, desc = self.queue_runner([])
                qsa_backend = runner._qwen_mrv2_qsa_backend
                target_builder = runner._qwen_mrv2_allowed_backend_builders[qsa_backend][0]
                runner._qwen_mrv2_allowed_backend_builders = None
                qsa_exports = NS(AscendQSABackend=qsa_backend, AscendQSAImpl=NS)
                if export is not None:
                    qsa_exports.AscendQSAStateMetadataBuilder = export
                modules = {
                    "vllm.models.qwen4_exp.amd.qsa": NS(Qwen4ExpQSAMetadataBuilder=target_builder),
                    "vllm.models.qwen4_exp.common.qsa_cache": NS(
                        QSAMetadataBuilder=StateBuilder, QSAStateBackend=StateBackend
                    ),
                    "vllm.v1.attention.backends.short_conv_attn": NS(
                        PleShortConvAttentionBackend=type("PleBackend", (), {}),
                        PleShortConvAttentionMetadataBuilder=type("PleBuilder", (), {}),
                    ),
                    "vllm_ascend.models.qwen4_exp": NS(qsa=qsa_exports),
                    "vllm_ascend.models.qwen4_exp.qsa": qsa_exports,
                    "vllm_ascend.ops.gdn_attn_builder": NS(
                        AscendGDNAttentionBackend=type("GDNBackend", (), {}),
                        AscendGDNAttentionMetadataBuilder=type("GDNBuilder", (), {}),
                        AscendGDNFusedAttentionBackend=type("GDNFused", (), {}),
                        AscendGDNHostMetadataBackend=type("GDNHost", (), {}),
                        AscendGDNHostMetadataBuilder=type("GDNHostBuilder", (), {}),
                    ),
                }
                runner.attn_groups[0].append(
                    NS(backend=StateBackend, get_metadata_builder=lambda _, builder=actual_builder: builder())
                )
                with patch.dict(sys.modules, modules):
                    self.assertEqual(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc), admitted)

    def test_mm_extension_requires_new_flag_and_exact_certificate(self):
        media = [
            NS(data=object(), modality="image", identifier="photo", mm_position=NS(offset=20, length=30, is_embed=None))
        ]
        runner, current_batch, output, desc = self.queue_runner(media)
        keys = runner._qwen_mrv2_mm_decode_request_keys(
            current_batch.req_ids, current_batch.idx_mapping_np, current_batch.prefill_len_np
        )
        runner._qwen_mrv2_mm_decode_state.record_snapshot(keys)
        self.assertTrue(runner._qwen_mrv2_mm_decode_state.certify(keys, [100]))
        self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
        runner._qwen_mrv2_mm_decode_queue_enabled = True
        self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
        self.assertEqual(runner._qwen_mrv2_mm_decode_stats["queue_eligible_steps"], 1)
        runner._qwen_mrv2_mm_decode_state.invalidate()
        self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))

    def test_text_and_certified_mm_queue_do_not_require_a_fixed_tp_size(self):
        for tp_size in (1, 2, 4, 8, 16, 32):
            for multimodal in (False, True):
                with self.subTest(tp_size=tp_size, multimodal=multimodal):
                    media = (
                        [
                            NS(
                                data=object(),
                                modality="image",
                                identifier="photo",
                                mm_position=NS(offset=20, length=30, is_embed=None),
                            )
                        ]
                        if multimodal
                        else []
                    )
                    runner, current_batch, output, desc = self.queue_runner(media)
                    runner.parallel_config.tensor_parallel_size = tp_size
                    runner._qwen_mrv2_mm_decode_queue_enabled = True
                    if multimodal:
                        # A different TP degree must not bypass prompt completion.
                        self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                        keys = runner._qwen_mrv2_mm_decode_request_keys(
                            current_batch.req_ids, current_batch.idx_mapping_np, current_batch.prefill_len_np
                        )
                        runner._qwen_mrv2_mm_decode_state.record_snapshot(keys)
                        self.assertTrue(runner._qwen_mrv2_mm_decode_state.certify(keys, [100]))
                    self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                    self.assertEqual(runner._qwen_mrv2_early_queue_last_guard["failed"], [])
                    runner._qwen_mrv2_early_queue_stats.update(
                        host_length_wait_skipped=0, length_device_wait_enqueued=0
                    )
                    runner.req_states.num_computed_tokens_cpu = np.array([104])
                    runner.input_buffers = NS(seq_lens_cpu=np.zeros(1, dtype=np.int32))
                    runner.num_computed_tokens_event = NS(
                        synchronize=lambda: self.fail("eligible decode synchronized the CPU")
                    )
                    waits = []
                    with patch.object(
                        namespace["torch"].npu, "current_stream", return_value=NS(wait_event=waits.append)
                    ):
                        runner._update_seq_lens_cpu(output, current_batch.req_ids, allow_cpu_upper_bound=True)
                    self.assertEqual(waits, [runner.num_computed_tokens_event])
                    self.assertEqual(runner.input_buffers.seq_lens_cpu.tolist(), [108])
                    self.assertEqual(runner._qwen_mrv2_early_queue_stats["host_length_wait_skipped"], 1)

    def test_cache_modes_scheduling_and_static_mtp_lengths_allow_early_queue(self):
        for k, prefix, cache_mode, async_scheduling, multimodal in product(
            (1, 2, 3, 4, 5, 8), (False, True), ("none", "align", "all"), (False, True), (False, True)
        ):
            with self.subTest(k=k, prefix=prefix, cache_mode=cache_mode, async_=async_scheduling, mm=multimodal):
                media = (
                    [
                        NS(
                            data=object(),
                            modality="image",
                            identifier="photo",
                            mm_position=NS(offset=20, length=30, is_embed=None),
                        )
                    ]
                    if multimodal
                    else []
                )
                runner, current_batch, output, desc = self.queue_runner(media, draft_tokens=k)
                runner.cache_config = NS(
                    enable_prefix_caching=prefix, mamba_cache_mode=cache_mode, kv_sharing_fast_prefill=True
                )
                runner.scheduler_config.async_scheduling = async_scheduling
                runner._qwen_mrv2_mm_decode_queue_enabled = True
                if multimodal:
                    self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                    keys = runner._qwen_mrv2_mm_decode_request_keys(
                        current_batch.req_ids, current_batch.idx_mapping_np, current_batch.prefill_len_np
                    )
                    runner._qwen_mrv2_mm_decode_state.record_snapshot(keys)
                    self.assertTrue(runner._qwen_mrv2_mm_decode_state.certify(keys, [100]))
                self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                runner._qwen_mrv2_early_queue_stats.update(host_length_wait_skipped=0, length_device_wait_enqueued=0)
                runner.req_states.num_computed_tokens_cpu = np.array([104])
                runner.input_buffers = NS(seq_lens_cpu=np.zeros(1, dtype=np.int32))
                # No subscript support: an accidental read of the pending D2H fails.
                runner.num_computed_tokens_cpu = object()
                runner.num_computed_tokens_event = NS(synchronize=lambda: self.fail("CPU wait on stable decode"))
                waits = []
                with patch.object(namespace["torch"].npu, "current_stream", return_value=NS(wait_event=waits.append)):
                    runner._update_seq_lens_cpu(output, current_batch.req_ids, allow_cpu_upper_bound=True)
                self.assertEqual(waits, [runner.num_computed_tokens_event])
                self.assertEqual(runner.input_buffers.seq_lens_cpu.tolist(), [104 + k + 1])

    def test_prefix_cache_transitions_still_consume_the_exact_snapshot(self):
        for changes in (
            dict(scheduled_new_reqs=[object()]),
            dict(scheduled_cached_reqs=NS(req_ids=["image"], resumed_req_ids=["image"])),
            dict(preempted_req_ids={"image"}),
            dict(kv_cache_block_copies=[(1, 2)]),
        ):
            with self.subTest(changes=changes):
                runner, current_batch, output, desc = self.queue_runner([])
                runner.cache_config = NS(
                    enable_prefix_caching=True, mamba_cache_mode="align", kv_sharing_fast_prefill=True
                )
                runner.req_states.num_computed_tokens_cpu = np.array([104])
                runner.input_buffers = NS(seq_lens_cpu=np.zeros(1, dtype=np.int32))
                exact = make_runner(exact=101)
                runner.num_computed_tokens_cpu = exact.num_computed_tokens_cpu
                runner.num_computed_tokens_event = exact.num_computed_tokens_event
                vars(output).update(changes)
                self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                runner._update_seq_lens_cpu(output, current_batch.req_ids)
                self.assertTrue(exact.synchronized)
                self.assertEqual(runner.input_buffers.seq_lens_cpu.tolist(), [105])

    def test_dp_pp_sp_and_weight_offload_do_not_exclude_local_decode(self):
        for dp_size, pp_size, sp, multimodal in product((1, 2, 4), (1, 2, 4), (False, True), (False, True)):
            with self.subTest(dp=dp_size, pp=pp_size, sp=sp, multimodal=multimodal):
                media = (
                    [
                        NS(
                            data=object(),
                            modality="image",
                            identifier="photo",
                            mm_position=NS(offset=20, length=30, is_embed=None),
                        )
                    ]
                    if multimodal
                    else []
                )
                runner, current_batch, output, desc = self.queue_runner(media)
                runner.parallel_config.data_parallel_size = dp_size
                runner.parallel_config.pipeline_parallel_size = pp_size
                runner.parallel_config.use_sequence_parallel_moe = sp
                runner.ascend_config.enable_sp_by_pass = sp
                runner.use_pp = pp_size > 1
                runner.is_last_pp_rank = True
                # PLE offload is separate from general UVA/prefetch offload.
                # Neither owns an exact CPU request-length consumer.
                runner.vllm_config.engram_config = NS(cpu_offload=True)
                runner.vllm_config.offload_config.uva = NS(cpu_offload_gb=8, cpu_offload_params=["embedding"])
                runner.vllm_config.offload_config.prefetch = NS(offload_group_size=2, offload_params=["weights"])
                # Other DP replicas may choose a larger graph. Local request
                # rows remain unpadded; padding is prepared after admission.
                desc.num_tokens = 16
                desc.num_reqs = 4
                runner._qwen_mrv2_mm_decode_queue_enabled = True
                if multimodal:
                    self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                    keys = runner._qwen_mrv2_mm_decode_request_keys(
                        current_batch.req_ids, current_batch.idx_mapping_np, current_batch.prefill_len_np
                    )
                    runner._qwen_mrv2_mm_decode_state.record_snapshot(keys)
                    self.assertTrue(runner._qwen_mrv2_mm_decode_state.certify(keys, [100]))
                self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                runner._qwen_mrv2_early_queue_stats.update(host_length_wait_skipped=0, length_device_wait_enqueued=0)
                runner.req_states.num_computed_tokens_cpu = np.array([104])
                runner.input_buffers = NS(seq_lens_cpu=np.zeros(1, dtype=np.int32))
                runner.num_computed_tokens_cpu = object()  # Must not read the pending pinned D2H.
                runner.num_computed_tokens_event = NS(synchronize=lambda: self.fail("unexpected host wait"))
                waits = []
                with patch.object(namespace["torch"].npu, "current_stream", return_value=NS(wait_event=waits.append)):
                    runner._update_seq_lens_cpu(output, current_batch.req_ids, allow_cpu_upper_bound=True)
                self.assertEqual(waits, [runner.num_computed_tokens_event])
                self.assertEqual(runner.input_buffers.seq_lens_cpu.tolist(), [108])

    def test_dp_empty_or_malformed_local_rows_fall_back_without_error(self):
        for case in (
            "empty",
            "padded_logical_rows",
            "different_local_schedule",
            "missing_slot",
            "negative_slot",
            "out_of_range_slot",
            "non_integer_slot",
            "wrong_local_token_count",
            "too_small_graph",
            "too_few_graph_rows",
            "inconsistent_graph_rows",
            "nonuniform_graph_capacity",
            "unknown_descriptor_schema",
        ):
            with self.subTest(case=case):
                runner, current_batch, output, desc = self.queue_runner([])
                runner.parallel_config.data_parallel_size = 4
                desc.num_tokens, desc.num_reqs = 16, 4
                if case == "empty":
                    current_batch.req_ids = []
                elif case == "padded_logical_rows":
                    current_batch.num_draft_tokens_np = np.array([3, -1])
                elif case == "different_local_schedule":
                    output.num_scheduled_tokens["image"] = 1
                elif case == "missing_slot":
                    runner.req_states.req_id_to_index = {}
                elif case == "negative_slot":
                    current_batch.idx_mapping_np = np.array([-1])
                elif case == "out_of_range_slot":
                    current_batch.idx_mapping_np = np.array([1])
                elif case == "non_integer_slot":
                    current_batch.idx_mapping_np = np.array([0.0])
                elif case == "wrong_local_token_count":
                    current_batch.num_tokens = 8
                elif case == "too_small_graph":
                    desc.num_tokens = 0
                elif case == "too_few_graph_rows":
                    desc.num_reqs = 0
                elif case == "inconsistent_graph_rows":
                    desc.num_reqs = 3
                elif case == "nonuniform_graph_capacity":
                    desc.num_tokens = 15
                else:
                    del desc.num_reqs
                self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                self.assertEqual(runner._qwen_mrv2_early_queue_last_guard["failed"], ["full_uniform_verification"])

    def test_dp_local_request_order_is_independent_of_global_graph_padding(self):
        runner, current_batch, output, desc = self.queue_runner([])
        runner.parallel_config.data_parallel_size = 4
        runner.req_states.req_id_to_index = {"left": 3, "right": 1}
        runner.req_states.num_computed_tokens_np = np.array([0, 104, 0, 204])
        runner.req_states.num_computed_tokens_cpu = np.array([0, 104, 0, 204])
        runner.encoder_cache.mm_features = {"left": [], "right": []}
        current_batch.req_ids = ["left", "right"]
        current_batch.idx_mapping_np = np.array([3, 1])
        current_batch.num_scheduled_tokens = np.array([4, 4])
        current_batch.num_draft_tokens_np = np.array([3, 3])
        current_batch.prefill_len_np = np.array([100, 100])
        current_batch.is_prefilling_np = np.array([False, False])
        current_batch.num_tokens = 8
        output.num_scheduled_tokens = {"left": 4, "right": 4}
        output.scheduled_spec_decode_tokens = {"left": [1, 2, 3], "right": [4, 5, 6]}
        desc.num_tokens, desc.num_reqs = 16, 4
        self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
        runner._qwen_mrv2_early_queue_stats.update(host_length_wait_skipped=0, length_device_wait_enqueued=0)
        runner.input_buffers = NS(seq_lens_cpu=np.zeros(2, dtype=np.int32))
        runner.num_computed_tokens_cpu = object()
        runner.num_computed_tokens_event = NS(synchronize=lambda: self.fail("unexpected host wait"))
        waits = []
        with patch.object(namespace["torch"].npu, "current_stream", return_value=NS(wait_event=waits.append)):
            runner._update_seq_lens_cpu(output, current_batch.req_ids, allow_cpu_upper_bound=True)
        self.assertEqual(runner.input_buffers.seq_lens_cpu.tolist(), [208, 108])
        self.assertEqual(waits, [runner.num_computed_tokens_event])
        # Do not admit a rank-local row mapping that aliases two real requests.
        current_batch.idx_mapping_np = np.array([3, 3])
        self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))

    def test_pp_stage_without_local_speculator_keeps_original_count_path(self):
        for exact_cpu_consumer in (False, True):
            with self.subTest(exact_cpu_consumer=exact_cpu_consumer):
                runner, current_batch, output, desc = self.queue_runner([])
                runner.parallel_config.pipeline_parallel_size = 4
                runner.use_pp = True
                runner.is_last_pp_rank = False
                runner.speculator = None
                runner.sync_spec_pp_cpu_counts = exact_cpu_consumer
                self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                self.assertEqual(
                    runner._qwen_mrv2_early_queue_last_guard["failed"], ["exact_qwen_model_state_and_mtp_speculator"]
                )
                exact = make_runner(exact=101)
                runner.num_computed_tokens_cpu = exact.num_computed_tokens_cpu
                runner.num_computed_tokens_event = exact.num_computed_tokens_event
                runner.req_states.num_computed_tokens_cpu = np.array([104])
                runner.input_buffers = NS(seq_lens_cpu=np.zeros(1, dtype=np.int32))
                runner._update_seq_lens_cpu(output, current_batch.req_ids)
                self.assertEqual(exact.synchronized, exact_cpu_consumer)
                self.assertEqual(runner.input_buffers.seq_lens_cpu.tolist(), [105 if exact_cpu_consumer else 108])

    def test_configured_mtp_lengths_still_reject_mismatched_batches_and_bounds(self):
        for k in (1, 2, 3, 4, 5, 8):
            for case in ("descriptor", "scheduled", "draft_counts", "draft_ids", "draft_limit"):
                with self.subTest(k=k, case=case):
                    runner, current_batch, output, desc = self.queue_runner([], draft_tokens=k)
                    runner.cache_config.enable_prefix_caching = True
                    runner.cache_config.mamba_cache_mode = "align"
                    runner.scheduler_config.async_scheduling = False
                    self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                    expected_guard = "full_uniform_verification"
                    if case == "descriptor":
                        desc.uniform_token_count = k + 2
                    elif case == "scheduled":
                        current_batch.num_scheduled_tokens = np.array([k + 2])
                    elif case == "draft_counts":
                        current_batch.num_draft_tokens_np = np.array([k + 1])
                    elif case == "draft_ids":
                        output.scheduled_spec_decode_tokens["image"] = list(range(k + 1))
                    else:
                        runner.speculator.draft_model_config.max_model_len = 104 + 2 * k
                        expected_guard = "bounds_within_model_and_drafter"
                    self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                    self.assertIn(expected_guard, runner._qwen_mrv2_early_queue_last_guard["failed"])

    def test_certified_mm_still_obeys_original_queue_guards(self):
        cases = (
            ("runner.parallel_config", "decode_context_parallel_size", 2, "dcp1_pcp1"),
            ("runner.parallel_config", "prefill_context_parallel_size", 2, "dcp1_pcp1"),
            ("runner", "num_speculative_steps", 0, "full_uniform_verification"),
            ("runner.speculative_config", "method", "eagle", "fixed_mtp"),
            ("runner.speculative_config", "uses_dynamic_speculative_decoding", lambda: True, "fixed_mtp"),
            ("desc", "cg_mode", object(), "full_uniform_verification"),
            ("desc", "uniform_token_count", 8, "full_uniform_verification"),
            ("batch", "num_scheduled_tokens", np.array([3]), "full_uniform_verification"),
            ("batch", "num_draft_tokens_np", np.array([2]), "full_uniform_verification"),
            ("output", "scheduled_spec_decode_tokens", {"image": [1, 2]}, "full_uniform_verification"),
            ("batch", "has_prefill", True, "pure_decode"),
            ("batch", "is_prefilling_np", np.array([True]), "pure_decode"),
            ("batch", "prefill_runs_as_decode_np", np.array([True]), "pure_decode"),
            ("output", "scheduled_new_reqs", [object()], "no_request_arrival_resume_or_finish"),
            ("output", "preempted_req_ids", {"image"}, "no_request_arrival_resume_or_finish"),
            ("output", "finished_req_ids", {"image"}, "no_request_arrival_resume_or_finish"),
            ("output.scheduled_cached_reqs", "resumed_req_ids", ["image"], "no_request_arrival_resume_or_finish"),
            ("runner", "max_model_len", 107, "bounds_within_model_and_drafter"),
            ("runner.speculator.draft_model_config", "max_model_len", 110, "bounds_within_model_and_drafter"),
            ("runner.model_state", "decoder_replay", object(), "no_decoder_replay"),
            ("runner.model_config", "enable_prompt_embeds", True, "text_without_prompt_embeddings"),
            ("output", "scheduled_encoder_inputs", {"image": [0]}, "text_without_prompt_embeddings"),
            ("runner.parallel_config", "enable_dbo", True, "no_dbo"),
            ("runner", "lora_config", object(), "no_lora"),
            ("runner", "adaptive_verification", object(), "no_connectors_or_adaptive_verification"),
            ("runner", "aux_output_connector", object(), "no_connectors_or_adaptive_verification"),
            ("runner", "sparse_kv_offload_manager", object(), "no_connectors_or_adaptive_verification"),
            ("output", "kv_connector_metadata", object(), "no_connectors_or_adaptive_verification"),
            ("output", "ec_connector_metadata", object(), "no_connectors_or_adaptive_verification"),
            ("output", "ec_manager_metadata", object(), "no_connectors_or_adaptive_verification"),
            ("output", "kv_cache_block_copies", [object()], "no_connectors_or_adaptive_verification"),
            ("output", "has_structured_output_requests", True, "no_connectors_or_adaptive_verification"),
            ("runner", "_qwen_mrv2_snapshot_generation", 0, "recorded_same_stream_snapshot"),
            ("runner", "_qwen_mrv2_snapshot_source_stream", "different-stream", "recorded_same_stream_snapshot"),
            ("runner", "sync_spec_pp_cpu_counts", True, "recorded_same_stream_snapshot"),
            ("runner", "_can_skip_spec_cpu_length_sync", lambda: True, "recorded_same_stream_snapshot"),
            ("runner.speculator", "attn_groups", [], "audited_backend_builder_pairs"),
            ("runner.speculator", "attn_architecture", "unknown", "audited_backend_builder_pairs"),
        )
        for tp_size, (path, name, value, expected_guard) in product((1, 2, 4, 8, 16, 32), cases):
            with self.subTest(tp_size=tp_size, path=path, name=name, guard=expected_guard):
                media = [
                    NS(
                        data=object(),
                        modality="image",
                        identifier="photo",
                        mm_position=NS(offset=20, length=30, is_embed=None),
                    )
                ]
                runner, current_batch, output, desc = self.queue_runner(media)
                runner.parallel_config.tensor_parallel_size = tp_size
                keys = runner._qwen_mrv2_mm_decode_request_keys(
                    current_batch.req_ids, current_batch.idx_mapping_np, current_batch.prefill_len_np
                )
                runner._qwen_mrv2_mm_decode_state.record_snapshot(keys)
                self.assertTrue(runner._qwen_mrv2_mm_decode_state.certify(keys, [100]))
                runner._qwen_mrv2_mm_decode_queue_enabled = True
                # Establish that the certificate admits the unmodified fixture.
                self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                runner._qwen_mrv2_mm_decode_stats["queue_eligible_steps"] = 0
                objects = {"runner": runner, "batch": current_batch, "output": output, "desc": desc}
                root_name, *attributes = path.split(".")
                obj = objects[root_name]
                for attribute in attributes:
                    obj = getattr(obj, attribute)
                setattr(obj, name, value)
                self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
                self.assertIn(expected_guard, runner._qwen_mrv2_early_queue_last_guard["failed"])
                self.assertEqual(runner._qwen_mrv2_mm_decode_stats["queue_eligible_steps"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
