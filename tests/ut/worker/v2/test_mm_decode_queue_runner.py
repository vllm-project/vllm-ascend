# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU behavior tests for the production snapshot and admission methods."""

import ast
import importlib.util
import sys
import unittest
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


class BaseRunner:
    def finish_requests(self, output):
        self.lifecycle_log.append("base_finish")


METHODS = {
    "finish_requests",
    "_qwen_mrv2_mm_decode_request_keys",
    "_qwen_mrv2_mm_decode_context",
    "_qwen_mrv2_can_queue_decode",
    "_qwen_mrv2_mm_batch_is_stable_decode",
    "_qwen_mrv2_mm_decode_is_certified",
    "_qwen_mrv2_certify_mm_decode",
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
    "_QWEN_MRV2_EARLY_QUEUE_DRAFT_TOKENS": 3,
    "_QWEN_MRV2_EARLY_QUEUE_TP_SIZE": 8,
    "CUDAGraphMode": NS(FULL_DECODE_ONLY=object(), FULL=object()),
    "has_kv_transfer_group": lambda: False,
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

    def queue_runner(self, media):
        runner = self.queue_environment_runner()
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
        current_batch.num_scheduled_tokens = np.array([4])
        current_batch.num_draft_tokens_np = np.array([3])
        output = scheduler(
            scheduled_spec_decode_tokens={"image": [1, 2, 3]},
            has_structured_output_requests=False,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
            ec_manager_metadata=None,
            kv_cache_block_copies=[],
        )
        desc = NS(cg_mode=namespace["CUDAGraphMode"].FULL, uniform_token_count=4)
        return runner, current_batch, output, desc

    def test_existing_text_queue_flag_behavior_is_unchanged(self):
        runner, current_batch, output, desc = self.queue_runner([])
        runner._qwen_mrv2_mm_decode_proof_enabled = False
        self.assertTrue(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))
        self.assertEqual(runner._qwen_mrv2_mm_decode_stats["queue_eligible_steps"], 0)
        runner._qwen_mrv2_early_queue_enabled = False
        self.assertFalse(runner._qwen_mrv2_can_queue_decode(output, current_batch, desc))

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

    def test_certified_mm_still_obeys_original_queue_guards(self):
        cases = (
            ("runner.parallel_config", "tensor_parallel_size", 4, "tp8_dp1_pp1_dcp1_pcp1"),
            ("runner.parallel_config", "data_parallel_size", 2, "tp8_dp1_pp1_dcp1_pcp1"),
            ("runner.parallel_config", "pipeline_parallel_size", 2, "tp8_dp1_pp1_dcp1_pcp1"),
            ("runner.parallel_config", "decode_context_parallel_size", 2, "tp8_dp1_pp1_dcp1_pcp1"),
            ("runner.parallel_config", "prefill_context_parallel_size", 2, "tp8_dp1_pp1_dcp1_pcp1"),
            ("runner.scheduler_config", "async_scheduling", False, "async_scheduling"),
            ("runner", "num_speculative_steps", 2, "fixed_k3"),
            ("runner.speculative_config", "method", "eagle", "fixed_k3"),
            ("runner.speculative_config", "uses_dynamic_speculative_decoding", lambda: True, "fixed_k3"),
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
            ("runner.cache_config", "enable_prefix_caching", True, "prefix_and_mamba_cache_disabled"),
            ("runner.cache_config", "mamba_cache_mode", "all", "prefix_and_mamba_cache_disabled"),
            ("runner.cache_config", "kv_sharing_fast_prefill", True, "prefix_and_mamba_cache_disabled"),
            ("runner.model_state", "decoder_replay", object(), "prefix_and_mamba_cache_disabled"),
            ("runner.model_config", "enable_prompt_embeds", True, "text_without_prompt_embeddings"),
            ("output", "scheduled_encoder_inputs", {"image": [0]}, "text_without_prompt_embeddings"),
            ("runner.parallel_config", "use_sequence_parallel_moe", True, "no_flashcomm_or_dbo"),
            ("runner.parallel_config", "enable_dbo", True, "no_flashcomm_or_dbo"),
            ("runner.ascend_config", "enable_sp_by_pass", True, "no_flashcomm_or_dbo"),
            ("runner", "lora_config", object(), "no_lora_or_weight_offload"),
            ("runner.vllm_config.offload_config.uva", "cpu_offload_gb", 1, "no_lora_or_weight_offload"),
            ("runner.vllm_config.offload_config.prefetch", "offload_group_size", 1, "no_lora_or_weight_offload"),
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
        for path, name, value, expected_guard in cases:
            with self.subTest(path=path, name=name, guard=expected_guard):
                media = [
                    NS(
                        data=object(),
                        modality="image",
                        identifier="photo",
                        mm_position=NS(offset=20, length=30, is_embed=None),
                    )
                ]
                runner, current_batch, output, desc = self.queue_runner(media)
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
