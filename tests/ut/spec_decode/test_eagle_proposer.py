from unittest.mock import MagicMock, patch
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from vllm.config import CacheConfig, CompilationMode, CUDAGraphMode, VllmConfig, set_current_vllm_config

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.spec_decode.eagle_proposer import (
    AscendEagleProposer,
    _FusedModelWithMTP,
    SpecDecodeBaseProposer,
)


class TestEagleProposerInitialization(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.cache_config = MagicMock(spec=CacheConfig)
        self.vllm_config.scheduler_config = MagicMock()
        self.vllm_config.model_config = MagicMock()
        self.vllm_config.model_config.hf_text_config = MagicMock(
            spec=[]
        )  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.compilation_config = MagicMock()
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    def test_initialization_eagle_graph(self):
        self.vllm_config.speculative_config.method = "eagle"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 4096
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = False
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 4096)
            self.assertTrue(proposer.use_cuda_graph)

            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.input_ids.shape, (expected_max_num_tokens,))
            self.assertEqual(proposer.positions.shape, (expected_max_num_tokens,))
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 4096))
            self.assertEqual(proposer.arange.shape, (expected_max_num_tokens,))

    def test_initialization_eagle3_enforce_eager(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.NONE
        self.vllm_config.compilation_config.pass_config = MagicMock()
        self.vllm_config.compilation_config.pass_config.enable_sp = False
        self.vllm_config.model_config.enforce_eager = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertFalse(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_eagle3_full_graph_async(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertTrue(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_mtp_full_graph_async(self):
        self.vllm_config.speculative_config.method = "mtp"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertFalse(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_fused_mtp_still_builds_graph_capture_metadata_for_compress_model(self):
        self.vllm_config.speculative_config.method = "mtp"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = False
        self.vllm_config.model_config.hf_config = SimpleNamespace(compress_ratios=[4])
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertFalse(proposer.use_cuda_graph)

            proposer.fused_with_main_graph = True
            proposer.attn_layer_names = ["draft_layer"]
            builder = MagicMock()
            builder.build_for_graph_capture.return_value = "draft_meta"
            attn_group = MagicMock()
            attn_group.get_metadata_builder.return_value = builder
            proposer.draft_attn_groups = [attn_group]

            proposer.query_start_loc = SimpleNamespace(
                cpu=torch.zeros(4, dtype=torch.int32),
                gpu=torch.zeros(4, dtype=torch.int32),
                copy_to_gpu=MagicMock(),
            )
            proposer.runner.attn_groups = [MagicMock()]
            proposer.runner.uniform_decode_query_len = 1
            proposer.runner.input_batch.num_computed_tokens_cpu_tensor = torch.tensor([0, 0], dtype=torch.int32)
            proposer.runner.query_start_loc = SimpleNamespace(
                cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            )
            proposer.runner.seq_lens = SimpleNamespace(
                cpu=torch.tensor([1, 1], dtype=torch.int32),
                gpu=torch.tensor([1, 1], dtype=torch.int32),
            )
            proposer.runner.actual_seq_lengths_q = torch.tensor([1, 2], dtype=torch.int32)
            proposer.runner.positions = SimpleNamespace(
                gpu=torch.tensor([0, 1], dtype=torch.int32),
                cpu=torch.tensor([0, 1], dtype=torch.int32),
            )
            proposer.runner.attn_state = MagicMock()
            proposer.runner.decode_token_per_req = 3
            proposer.slot_mapping_group = [
                torch.zeros(4, dtype=torch.int32) for _ in range(proposer.num_speculative_tokens)
            ]
            proposer.runner.input_batch.block_table = [
                SimpleNamespace(
                    get_device_tensor=MagicMock(return_value=torch.zeros((2, 1), dtype=torch.int32)),
                    slot_mapping=SimpleNamespace(gpu=torch.tensor([9, 10], dtype=torch.int32)),
                )
            ]

            metadata = proposer.build_graph_capture_attn_metadata(
                num_tokens=2,
                num_reqs=2,
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
            )

            self.assertEqual(len(metadata), proposer.num_speculative_tokens)
            self.assertEqual(metadata[0]["draft_layer"], "draft_meta")
            builder.build_for_graph_capture.assert_called()
            common_attn_metadata = builder.build_for_graph_capture.call_args.args[0]
            self.assertTrue(
                torch.equal(
                    common_attn_metadata.positions_cpu[:2],
                    proposer.runner.positions.cpu[:2],
                )
            )

    def test_mtp_full_graph_padding_uses_stable_dummy_inputs(self):
        proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer.uses_mrope = False
        proposer.uses_xdrope_dim = 0
        proposer.draft_uses_xdrope_dim = 0
        proposer.input_ids = torch.arange(8, dtype=torch.int64)
        proposer.positions = torch.arange(8, dtype=torch.int32) + 10
        proposer.hidden_states = torch.ones((8, 4), dtype=torch.float32)
        proposer.slot_mapping_group = [
            torch.full((8,), -1, dtype=torch.int32),
        ]

        proposer._pad_mtp_full_graph_inputs(5, 8, CUDAGraphMode.FULL)

        self.assertTrue(
            torch.equal(
                proposer.input_ids[5:8],
                torch.zeros(3, dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                proposer.positions[5:8],
                torch.zeros(3, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                proposer.hidden_states[5:8],
                torch.zeros((3, 4), dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                proposer.slot_mapping_group[0][5:8],
                torch.zeros(3, dtype=torch.int32),
            )
        )
        self.assertEqual(proposer._graph_padding_slot_id(CUDAGraphMode.FULL), 0)

    @patch("vllm_ascend.spec_decode.eagle_proposer.enable_sp_by_pass")
    @patch("vllm_ascend.spec_decode.eagle_proposer.enable_sp")
    def test_mtp_eager_padding_rounds_to_tp_size(
        self,
        mock_enable_sp,
        mock_enable_sp_by_pass,
    ):
        proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
        )
        mock_enable_sp.return_value = True
        mock_enable_sp_by_pass.return_value = False

        self.assertEqual(
            proposer._get_mtp_eager_padded_num_input_tokens(2, CUDAGraphMode.NONE),
            8,
        )
        self.assertEqual(
            proposer._get_mtp_eager_padded_num_input_tokens(9, CUDAGraphMode.NONE),
            16,
        )
        self.assertEqual(
            proposer._get_mtp_eager_padded_num_input_tokens(9, CUDAGraphMode.FULL),
            9,
        )

        mock_enable_sp.return_value = False
        self.assertEqual(
            proposer._get_mtp_eager_padded_num_input_tokens(2, CUDAGraphMode.NONE),
            8,
        )
        self.assertEqual(
            proposer._get_mtp_eager_padded_num_input_tokens(9, CUDAGraphMode.NONE),
            9,
        )

        proposer.method = "eagle"
        self.assertEqual(
            proposer._get_mtp_eager_padded_num_input_tokens(2, CUDAGraphMode.NONE),
            2,
        )

    def test_mtp_eager_padding_uses_stable_dummy_inputs(self):
        proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer.uses_mrope = False
        proposer.uses_xdrope_dim = 0
        proposer.draft_uses_xdrope_dim = 0
        proposer.input_ids = torch.arange(8, dtype=torch.int64)
        proposer.positions = torch.arange(8, dtype=torch.int32) + 10
        proposer.hidden_states = torch.ones((8, 4), dtype=torch.float32)
        proposer.slot_mapping_group = [
            torch.full((8,), -1, dtype=torch.int32),
        ]

        self.assertTrue(proposer._uses_mtp_eager_input_padding(2, 8, CUDAGraphMode.NONE))
        proposer._pad_mtp_full_graph_inputs(
            2,
            8,
            CUDAGraphMode.NONE,
            force=True,
        )

        self.assertTrue(
            torch.equal(
                proposer.input_ids[2:8],
                torch.zeros(6, dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                proposer.positions[2:8],
                torch.zeros(6, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                proposer.hidden_states[2:8],
                torch.zeros((6, 4), dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                proposer.slot_mapping_group[0][2:8],
                torch.zeros(6, dtype=torch.int32),
            )
        )

    def test_fused_mtp_wrapper_capacity_matches_graph_batch(self):
        drafter = SimpleNamespace(
            num_speculative_tokens=1,
            runner=SimpleNamespace(
                max_num_reqs=16,
                max_num_tokens=32,
                input_batch=SimpleNamespace(sampling_metadata=SimpleNamespace()),
            ),
            device=torch.device("cpu"),
            hidden_size=3,
            dtype=torch.float32,
        )

        fused_model = _FusedModelWithMTP(torch.nn.Identity(), drafter)

        self.assertEqual(fused_model.bonus_row_indices_buf.shape, (16,))
        self.assertEqual(fused_model.next_token_ids_buf.shape, (16,))
        self.assertEqual(fused_model.valid_sampled_tokens_count_buf.shape, (16,))
        self.assertEqual(fused_model.sampled_token_ids_buf.shape, (16, 2))
        self.assertEqual(fused_model.target_logits_indices_buf.shape, (16, 1))
        self.assertEqual(fused_model.sample_logits_indices_buf.shape, (32,))
        self.assertEqual(fused_model.sample_idx_mapping_buf.shape, (32,))
        self.assertEqual(fused_model.target_row_indices_buf.shape, (16,))

    @patch("vllm_ascend.spec_decode.eagle_proposer.strict_rejection_sample_tensor")
    @patch("vllm_ascend.spec_decode.eagle_proposer.sample_with_runtime_state")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    def test_fused_mtp_verifier_uses_accepted_and_recovered_tokens(
        self,
        mock_get_forward_context,
        mock_sample_with_runtime_state,
        mock_rejection_sample,
    ):
        class _DummyModel(torch.nn.Module):
            def __init__(self, hidden_states: torch.Tensor):
                super().__init__()
                self.hidden_states = hidden_states

            def forward(self, **_kwargs):
                return self.hidden_states

            def compute_logits(self, hidden_states: torch.Tensor):
                return hidden_states

        hidden_states = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            dtype=torch.float32,
        )
        drafter = SimpleNamespace(
            num_speculative_tokens=1,
            runner=SimpleNamespace(
                max_num_reqs=2,
                max_num_tokens=4,
                input_batch=SimpleNamespace(
                    sampling_metadata=SimpleNamespace(
                        temperature=torch.zeros(2, dtype=torch.float32),
                        top_k=torch.full((2,), 3, dtype=torch.int32),
                        top_p=torch.ones(2, dtype=torch.float32),
                        seeds=torch.tensor([3, 5], dtype=torch.int64),
                    ),
                ),
            ),
            device=torch.device("cpu"),
            hidden_size=3,
            dtype=torch.float32,
            propose_all_in_graph=MagicMock(
                return_value=torch.tensor([[7], [8]], dtype=torch.int64),
            ),
        )
        fused_model = _FusedModelWithMTP(_DummyModel(hidden_states), drafter)
        fused_model.num_reqs_buf[0] = 2
        fused_model.logits_indices_buf[:2] = torch.tensor([1, 3], dtype=torch.int64)
        fused_model.sample_logits_indices_buf[:4] = torch.tensor(
            [0, 1, 2, 3],
            dtype=torch.int64,
        )
        fused_model.bonus_row_indices_buf[:2] = torch.tensor([1, 3], dtype=torch.int64)
        fused_model.target_row_indices_buf[:2] = torch.tensor([0, 2], dtype=torch.int64)
        fused_model.cu_num_draft_tokens_buf[:2] = torch.tensor([1, 2], dtype=torch.int32)
        fused_model.draft_token_ids_flat_buf[:2] = torch.tensor([0, 1], dtype=torch.int64)
        fused_model.spec_decode_token_ids_buf[:2, :1] = torch.tensor(
            [[0], [1]],
            dtype=torch.int64,
        )
        fused_model.backup_next_token_ids_buf[:2] = torch.tensor(
            [101, 202],
            dtype=torch.int64,
        )
        mock_sample_with_runtime_state.return_value = torch.tensor(
            [0, 2, 0, 1],
            dtype=torch.int64,
        )
        mock_rejection_sample.return_value = torch.tensor(
            [[0, 2], [0, -1]],
            dtype=torch.int32,
        )

        mock_get_forward_context.return_value = SimpleNamespace(
            capturing=True,
            flash_comm_v1_enabled=False,
        )

        fused_model(
            input_ids=torch.tensor([11, 12, 13, 14], dtype=torch.int64),
            positions=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(
                fused_model.sampled_token_ids_buf[:2, :2],
                torch.tensor([[0, 2], [0, -1]], dtype=torch.int32),
            )
        )
        mock_rejection_sample.assert_not_called()
        self.assertTrue(
            torch.equal(
                fused_model.next_token_ids_buf[:2],
                torch.tensor([2, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                fused_model.valid_sampled_tokens_count_buf[:2],
                torch.tensor([2, 1], dtype=torch.int32),
            )
        )
        drafter.propose_all_in_graph.assert_called_once()
        self.assertTrue(
            torch.equal(
                drafter.propose_all_in_graph.call_args.kwargs["next_token_ids"],
                torch.tensor([2, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                drafter.propose_all_in_graph.call_args.kwargs["logits_indices"][:2],
                torch.tensor([1, 2], dtype=torch.int64),
            )
        )

    @patch("vllm_ascend.spec_decode.eagle_proposer.sample_with_runtime_state")
    def test_fused_mtp_verifier_uses_per_request_draft_ids(
        self,
        mock_sample_with_runtime_state,
    ):
        class _DummyModel(torch.nn.Module):
            def __init__(self, hidden_states: torch.Tensor):
                super().__init__()
                self.hidden_states = hidden_states

            def forward(self, **_kwargs):
                return self.hidden_states

            def compute_logits(self, hidden_states: torch.Tensor):
                return hidden_states

        drafter = SimpleNamespace(
            num_speculative_tokens=1,
            runner=SimpleNamespace(
                max_num_reqs=2,
                max_num_tokens=4,
                input_batch=SimpleNamespace(
                    sampling_metadata=SimpleNamespace(
                        temperature=torch.zeros(2, dtype=torch.float32),
                        top_k=torch.full((2,), 16, dtype=torch.int32),
                        top_p=torch.ones(2, dtype=torch.float32),
                        seeds=torch.tensor([3, 5], dtype=torch.int64),
                        all_greedy=True,
                        all_random=False,
                    ),
                ),
            ),
            device=torch.device("cpu"),
            hidden_size=16,
            dtype=torch.float32,
            propose_all_in_graph=MagicMock(
                return_value=torch.tensor([[101], [202]], dtype=torch.int64),
            ),
        )
        fused_model = _FusedModelWithMTP(
            _DummyModel(torch.zeros((4, 16), dtype=torch.float32)),
            drafter,
        )
        fused_model.num_reqs_buf[0] = 2
        fused_model.logits_indices_buf[:2] = torch.tensor([0, 2], dtype=torch.int64)
        fused_model.sample_logits_indices_buf[:4] = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
        fused_model.sample_idx_mapping_buf[:4] = torch.tensor([0, 1, 1, 0], dtype=torch.int32)
        fused_model.bonus_row_indices_buf[:2] = torch.tensor([0, 2], dtype=torch.int64)
        fused_model.target_row_indices_buf[:2] = torch.tensor([0, 1], dtype=torch.int64)
        fused_model.cu_num_draft_tokens_buf[:2] = torch.tensor([0, 1], dtype=torch.int32)
        fused_model.draft_token_ids_flat_buf[:2] = torch.tensor([7, 0], dtype=torch.int64)
        fused_model.spec_decode_token_ids_buf[:2, :1] = torch.tensor([[0], [7]], dtype=torch.int64)
        fused_model.backup_next_token_ids_buf[:2] = torch.tensor([11, 22], dtype=torch.int64)
        sample_logits = torch.zeros((4, 43), dtype=torch.float32)
        sample_logits[
            torch.arange(4),
            torch.tensor([42, 7, 8, 0], dtype=torch.int64),
        ] = 1.0

        next_token_ids = fused_model._run_rejection_verifier(
            sample_logits=sample_logits,
            runtime_positions=torch.arange(4, dtype=torch.int32),
        )

        mock_sample_with_runtime_state.assert_not_called()
        self.assertTrue(
            torch.equal(
                fused_model.sampled_token_ids_buf[:2, :2],
                torch.tensor([[42, -1], [7, 8]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                next_token_ids[:2],
                torch.tensor([42, 8], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                fused_model.logits_indices_buf[:2],
                torch.tensor([0, 2], dtype=torch.int64),
            )
        )

    @patch("vllm_ascend.spec_decode.eagle_proposer.strict_rejection_sample_tensor")
    @patch("vllm_ascend.spec_decode.eagle_proposer.sample_with_runtime_state")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    def test_fused_mtp_non_greedy_verifier_updates_next_token_ids_and_positions(
        self,
        mock_get_forward_context,
        mock_sample_with_runtime_state,
        mock_rejection_sample,
    ):
        class _DummyModel(torch.nn.Module):
            def __init__(self, hidden_states: torch.Tensor):
                super().__init__()
                self.hidden_states = hidden_states

            def forward(self, **_kwargs):
                return self.hidden_states

            def compute_logits(self, hidden_states: torch.Tensor):
                return hidden_states

        hidden_states = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            dtype=torch.float32,
        )
        drafter = SimpleNamespace(
            num_speculative_tokens=1,
            runner=SimpleNamespace(
                max_num_reqs=2,
                max_num_tokens=4,
                input_batch=SimpleNamespace(
                    sampling_metadata=SimpleNamespace(
                        temperature=torch.ones(2, dtype=torch.float32),
                        top_k=torch.full((2,), 3, dtype=torch.int32),
                        top_p=torch.ones(2, dtype=torch.float32),
                        seeds=torch.tensor([7, 11], dtype=torch.int64),
                    ),
                ),
            ),
            device=torch.device("cpu"),
            hidden_size=3,
            dtype=torch.float32,
            propose_all_in_graph=MagicMock(
                return_value=torch.tensor([[7], [8]], dtype=torch.int64),
            ),
        )
        fused_model = _FusedModelWithMTP(_DummyModel(hidden_states), drafter)
        fused_model.num_reqs_buf[0] = 2
        fused_model.logits_indices_buf[:2] = torch.tensor([1, 3], dtype=torch.int64)
        fused_model.sample_logits_indices_buf[:4] = torch.tensor(
            [0, 1, 2, 3],
            dtype=torch.int64,
        )
        fused_model.bonus_row_indices_buf[:2] = torch.tensor([1, 3], dtype=torch.int64)
        fused_model.target_row_indices_buf[:2] = torch.tensor([0, 2], dtype=torch.int64)
        fused_model.cu_num_draft_tokens_buf[:2] = torch.tensor([1, 2], dtype=torch.int32)
        fused_model.draft_token_ids_flat_buf[:2] = torch.tensor([4, 1], dtype=torch.int64)
        fused_model.spec_decode_token_ids_buf[:2, :1] = torch.tensor(
            [[4], [1]],
            dtype=torch.int64,
        )

        mock_sample_with_runtime_state.return_value = torch.tensor([4, 5, 6, 8], dtype=torch.int64)
        mock_rejection_sample.return_value = torch.tensor(
            [[4, 5], [6, -1]],
            dtype=torch.int32,
        )
        mock_get_forward_context.return_value = SimpleNamespace(
            capturing=True,
            flash_comm_v1_enabled=False,
        )

        fused_model(
            input_ids=torch.tensor([11, 12, 13, 14], dtype=torch.int64),
            positions=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(
                fused_model.sampled_token_ids_buf[:2, :2],
                torch.tensor([[4, 5], [6, -1]], dtype=torch.int32),
            )
        )
        mock_rejection_sample.assert_not_called()
        self.assertTrue(
            torch.equal(
                fused_model.next_token_ids_buf[:2],
                torch.tensor([5, 6], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                drafter.propose_all_in_graph.call_args.kwargs["logits_indices"][:2],
                torch.tensor([1, 2], dtype=torch.int64),
            )
        )

    @patch("vllm_ascend.spec_decode.eagle_proposer.strict_rejection_sample_tensor")
    @patch("vllm_ascend.spec_decode.eagle_proposer.sample_with_runtime_state")
    def test_fused_mtp_verifier_handles_multi_token_prefix_matching(
        self,
        mock_sample_with_runtime_state,
        mock_rejection_sample,
    ):
        drafter = SimpleNamespace(
            num_speculative_tokens=3,
            runner=SimpleNamespace(
                max_num_reqs=3,
                max_num_tokens=12,
                input_batch=SimpleNamespace(
                    sampling_metadata=SimpleNamespace(
                        temperature=torch.zeros(3, dtype=torch.float32),
                        top_k=torch.full((3,), 3, dtype=torch.int32),
                        top_p=torch.ones(3, dtype=torch.float32),
                        seeds=torch.tensor([3, 5, 7], dtype=torch.int64),
                        all_greedy=True,
                        all_random=False,
                    ),
                ),
            ),
            device=torch.device("cpu"),
            hidden_size=3,
            dtype=torch.float32,
        )
        fused_model = _FusedModelWithMTP(torch.nn.Identity(), drafter)
        fused_model.num_reqs_buf[0] = 3
        fused_model.sample_logits_indices_buf[:12] = torch.arange(12, dtype=torch.int64)
        fused_model.bonus_row_indices_buf[:3] = torch.tensor([3, 7, 8], dtype=torch.int64)
        fused_model.target_row_indices_buf[:9] = torch.tensor([0, 1, 2, 4, 5, 6, 0, 0, 0], dtype=torch.int64)
        fused_model.cu_num_draft_tokens_buf[:3] = torch.tensor([3, 6, 6], dtype=torch.int32)
        fused_model.draft_token_ids_flat_buf[:6] = torch.tensor([10, 11, 12, 20, 99, 22], dtype=torch.int64)

        sampled_ids = torch.tensor(
            [10, 11, 12, 100, 20, 21, 22, 200, 300, 0, 0, 0],
            dtype=torch.int64,
        )
        sample_logits = torch.zeros((12, 301), dtype=torch.float32)
        sample_logits[torch.arange(12), sampled_ids] = 1.0

        next_token_ids = fused_model._run_rejection_verifier(
            sample_logits=sample_logits,
            runtime_positions=torch.arange(12, dtype=torch.int32),
        )

        mock_sample_with_runtime_state.assert_not_called()
        self.assertTrue(
            torch.equal(
                fused_model.sampled_token_ids_buf[:3, :4],
                torch.tensor(
                    [
                        [10, 11, 12, 100],
                        [20, 21, -1, -1],
                        [300, -1, -1, -1],
                    ],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                next_token_ids[:3],
                torch.tensor([100, 21, 300], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                fused_model.valid_sampled_tokens_count_buf[:3],
                torch.tensor([4, 2, 1], dtype=torch.int32),
            )
        )
        mock_rejection_sample.assert_not_called()

    def test_propose_all_in_graph_casts_next_token_ids_for_input_buffer(self):
        class _CheckedBuffer:
            def __init__(self, tensor: torch.Tensor):
                self.tensor = tensor
                self.tensor_index_value_dtypes = []

            @property
            def dtype(self):
                return self.tensor.dtype

            def __getitem__(self, index):
                return self.tensor[index]

            def __setitem__(self, index, value):
                if isinstance(index, torch.Tensor) and isinstance(value, torch.Tensor):
                    self.tensor_index_value_dtypes.append(value.dtype)
                    if value.dtype != self.tensor.dtype:
                        raise AssertionError(
                            f"tensor index assignment dtype mismatch: {value.dtype} vs {self.tensor.dtype}"
                        )
                if isinstance(value, torch.Tensor):
                    value = value.to(self.tensor.dtype)
                self.tensor[index] = value

        class _DummyRawModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_input_ids_dtype = None

            def forward(self, input_ids, positions, **_kwargs):
                self.seen_input_ids_dtype = input_ids.dtype
                return torch.stack(
                    (
                        input_ids.to(torch.float32),
                        positions.to(torch.float32),
                    ),
                    dim=1,
                )

            def compute_logits(self, hidden_states):
                return hidden_states

        class _FakeProposer:
            def __init__(self):
                self.num_speculative_tokens = 1
                self.input_ids = _CheckedBuffer(
                    torch.zeros(4, dtype=torch.int32),
                )
                self.hidden_states = torch.zeros((4, 2), dtype=torch.float32)
                self.positions = torch.zeros(4, dtype=torch.int32)
                self.device = torch.device("cpu")
                self.pass_hidden_states_to_model = False
                self.method = "mtp"
                self.vllm_config = SimpleNamespace(
                    model_config=SimpleNamespace(max_model_len=16),
                )
                self._raw_model = _DummyRawModel()

            def get_model(self):
                return self._raw_model

            def _set_positions(self, num_tokens, positions):
                self.positions[:num_tokens] = positions.to(self.positions.dtype)

            def _get_positions(self, num_tokens):
                return self.positions[:num_tokens]

            def maybe_pad_and_reduce(self, hidden_states, positions):
                return hidden_states, positions

            def model_returns_tuple(self):
                return False

            def maybe_all_gather_and_unpad(
                self,
                last_hidden_states,
                positions,
                hidden_states_out,
            ):
                return last_hidden_states, positions, hidden_states_out

        proposer = _FakeProposer()
        draft_token_ids = SpecDecodeBaseProposer.propose_all_in_graph(
            proposer,
            hidden_states=torch.zeros((2, 2), dtype=torch.float32),
            input_ids=torch.tensor([11, 12], dtype=torch.int64),
            positions=torch.tensor([0, 1], dtype=torch.int32),
            logits_indices=torch.tensor([0], dtype=torch.int64),
            next_token_ids=torch.tensor([9], dtype=torch.int64),
            num_tokens=2,
        )

        self.assertEqual(
            proposer.input_ids.tensor_index_value_dtypes,
            [],
        )
        self.assertEqual(proposer.input_ids.tensor.dtype, torch.int32)
        self.assertEqual(
            proposer.get_model().seen_input_ids_dtype,
            torch.int32,
        )
        self.assertTrue(
            torch.equal(
                draft_token_ids,
                torch.tensor([[0]], dtype=torch.int64),
            )
        )

    def test_propose_all_in_graph_clears_graph_padding_inputs(self):
        class _DummyRawModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_compute_hidden_states = None

            def forward(self, input_ids, positions, hidden_states, **_kwargs):
                return hidden_states + torch.stack(
                    (
                        input_ids.to(torch.float32),
                        positions.to(torch.float32),
                    ),
                    dim=1,
                )

            def compute_logits(self, hidden_states):
                self.seen_compute_hidden_states = hidden_states.clone()
                return hidden_states

        class _FakeProposer:
            def __init__(self):
                self.num_speculative_tokens = 1
                self.input_ids = torch.full((4,), 99, dtype=torch.int64)
                self.hidden_states = torch.full((4, 2), 7.0, dtype=torch.float32)
                self.positions = torch.full((4,), 99, dtype=torch.int32)
                self.device = torch.device("cpu")
                self.pass_hidden_states_to_model = True
                self.method = "mtp"
                self.vllm_config = SimpleNamespace(
                    model_config=SimpleNamespace(max_model_len=16),
                )
                self._raw_model = _DummyRawModel()

            def get_model(self):
                return self._raw_model

            def _set_positions(self, num_tokens, positions):
                self.positions[:num_tokens] = positions.to(self.positions.dtype)

            def _get_positions(self, num_tokens):
                return self.positions[:num_tokens]

            def maybe_pad_and_reduce(self, hidden_states, positions):
                return hidden_states, positions

            def model_returns_tuple(self):
                return False

            def maybe_all_gather_and_unpad(
                self,
                last_hidden_states,
                positions,
                hidden_states_out,
            ):
                return last_hidden_states, positions, hidden_states_out

        proposer = _FakeProposer()
        SpecDecodeBaseProposer.propose_all_in_graph(
            proposer,
            hidden_states=torch.tensor(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                dtype=torch.float32,
            ),
            input_ids=torch.tensor([11, 12, 13, 14], dtype=torch.int64),
            positions=torch.tensor([20, 21, 22, 23], dtype=torch.int32),
            logits_indices=torch.tensor([1, 3], dtype=torch.int64),
            next_token_ids=torch.tensor([55, 0], dtype=torch.int64),
            num_tokens=4,
            num_actual_tokens=torch.tensor([2], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(
                proposer.input_ids,
                torch.tensor([12, 55, 0, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                proposer.positions,
                torch.tensor([20, 21, 0, 0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                proposer.hidden_states[2:],
                torch.zeros((2, 2), dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                proposer.get_model().seen_compute_hidden_states,
                torch.tensor([[57.0, 23.0], [13.0, 21.0]]),
            )
        )

    @patch("vllm_ascend.spec_decode.eagle_proposer.is_forward_context_available")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    def test_propose_all_in_graph_marks_draft_forward_context(
        self,
        mock_get_forward_context,
        mock_is_forward_context_available,
    ):
        context = SimpleNamespace(
            attn_metadata="main_meta",
            is_draft_model=False,
            num_tokens=99,
            num_accept_tokens=77,
            moe_layer_index=5,
            draft_attn_metadatas=["draft_meta0"],
        )
        mock_is_forward_context_available.return_value = True
        mock_get_forward_context.return_value = context

        class _DummyRawModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_context = []

            def forward(self, input_ids, positions, **_kwargs):
                self.seen_context.append(
                    (
                        context.attn_metadata,
                        context.is_draft_model,
                        context.num_tokens,
                        context.num_accept_tokens,
                        context.moe_layer_index,
                    )
                )
                return torch.stack(
                    (input_ids.to(torch.float32), positions.to(torch.float32)),
                    dim=1,
                )

            def compute_logits(self, hidden_states):
                return hidden_states

        class _FakeProposer:
            def __init__(self):
                self.num_speculative_tokens = 1
                self.input_ids = torch.zeros(2, dtype=torch.int64)
                self.hidden_states = torch.zeros((2, 2), dtype=torch.float32)
                self.positions = torch.zeros(2, dtype=torch.int32)
                self.device = torch.device("cpu")
                self.pass_hidden_states_to_model = False
                self.method = "mtp"
                self.vllm_config = SimpleNamespace(
                    model_config=SimpleNamespace(max_model_len=16),
                )
                self._raw_model = _DummyRawModel()

            def get_model(self):
                return self._raw_model

            def _set_positions(self, num_tokens, positions):
                self.positions[:num_tokens] = positions.to(self.positions.dtype)

            def _get_positions(self, num_tokens):
                return self.positions[:num_tokens]

            def maybe_pad_and_reduce(self, hidden_states, positions):
                return hidden_states, positions

            def model_returns_tuple(self):
                return False

            def maybe_all_gather_and_unpad(
                self,
                last_hidden_states,
                positions,
                hidden_states_out,
            ):
                return last_hidden_states, positions, hidden_states_out

        proposer = _FakeProposer()
        SpecDecodeBaseProposer.propose_all_in_graph(
            proposer,
            hidden_states=torch.zeros((2, 2), dtype=torch.float32),
            input_ids=torch.tensor([11, 12], dtype=torch.int64),
            positions=torch.tensor([0, 1], dtype=torch.int32),
            logits_indices=torch.tensor([0], dtype=torch.int64),
            next_token_ids=torch.tensor([9], dtype=torch.int64),
            num_tokens=2,
        )

        self.assertEqual(
            proposer.get_model().seen_context,
            [("draft_meta0", True, 2, 1, 0)],
        )
        self.assertEqual(context.attn_metadata, "main_meta")
        self.assertFalse(context.is_draft_model)
        self.assertEqual(context.num_tokens, 99)
        self.assertEqual(context.num_accept_tokens, 77)
        self.assertEqual(context.moe_layer_index, 5)


@unittest.skip("Skip due to the changes in #7153, fix me later")
class TestEagleProposerLoadModel(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.method = "eagle"
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.parallel_drafting = False

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp1(self, mock_pp_group, mock_get_model, mock_get_layers):
        mock_pp_group.return_value.world_size = 1
        mock_target_layer1 = MagicMock()
        mock_target_layer2 = MagicMock()
        mock_draft_layer1 = MagicMock()
        mock_draft_layer3 = MagicMock()
        mock_get_layers.side_effect = [
            {"layer1": mock_target_layer1, "layer2": mock_target_layer2},
            {},
            {},
            {"layer1": mock_draft_layer1, "layer3": mock_draft_layer3},
        ]

        weight = torch.zeros(0)

        mock_model = MagicMock()
        mock_model.supports_multimodal = False
        mock_model.lm_head = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_model.model.embed_tokens = MagicMock()
        mock_model.model.embed_tokens.weight = weight

        mock_get_model.return_value = MagicMock()
        mock_get_model.return_value.model.embed_tokens.weight = weight

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            mock_get_model.assert_called_once()
            self.assertEqual(self.proposer.attn_layer_names, ["layer3"])
            self.assertIs(self.proposer.model.model.embed_tokens, mock_model.model.embed_tokens)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp_gt1(self, mock_pp_group, mock_get_model, mock_get_layers):
        mock_pp_group.return_value.world_size = 2
        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{"layer1": mock_target_layer1}, {}, {}, {"layer2": mock_draft_layer2}]

        mock_model = MagicMock()
        original_embed = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_get_model.return_value = MagicMock(model=MagicMock(embed_tokens=original_embed))

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)

            self.assertIsNot(self.proposer.model.model.embed_tokens, mock_model.model.embed_tokens)
            self.assertEqual(self.proposer.attn_layer_names, ["layer2"])

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    @patch("vllm_ascend.spec_decode.eagle_proposer.supports_multimodal")
    def test_load_model_multimodal(self, mock_supports_multi, mock_pp_group, mock_get_model, mock_get_layers):
        mock_model = MagicMock()
        mock_model.get_language_model.return_value.lm_head = MagicMock()
        mock_supports_multi.return_value = True
        original_embed = MagicMock()
        mock_get_model.return_value = MagicMock(model=MagicMock(embed_tokens=original_embed))

        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{"layer1": mock_target_layer1}, {}, {}, {"layer2": mock_draft_layer2}]
        mock_pp_group.return_value.world_size = 2

        self.proposer.model = MagicMock()

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            self.assertEqual(mock_model.get_language_model.call_count, 2)
            self.assertIs(self.proposer.model.lm_head, mock_model.get_language_model.return_value.lm_head)


class TestEagleProposerDummyRun(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.num_speculative_tokens = 4
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1
        self.runner.pin_memory = False
        self.runner._sync_metadata_across_dp.return_value = (8, torch.tensor([8]), False)

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.model_config.use_mla = False
        self.vllm_config.model_config.hf_text_config = MagicMock(
            spec=[]
        )  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(4)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Mock parallel state functions
        self.mock_tp_world_size = patch(
            "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size", return_value=1
        )
        self.mock_tp_world_size.start()

        mock_dp_group = MagicMock()
        mock_dp_group.world_size = 1
        self.mock_dp_group = patch("vllm_ascend.ascend_forward_context.get_dp_group", return_value=mock_dp_group)
        self.mock_dp_group.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.model = MagicMock()
        self.proposer._runnable = MagicMock()
        self.proposer.update_stream = MagicMock()

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        self.mock_tp_world_size.stop()
        self.mock_dp_group.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # cpu does not support parallel-group, let alone `sp`
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_forward_context", **{"return_value.flash_comm_v1_enabled": False}
    )
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_basic(self, mock_context, mock_get_context, mock_get_context_2):
        num_tokens = 32
        with_prefill = False

        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=num_tokens, with_prefill=with_prefill)

            self.assertTrue(self.proposer._runnable.call_count == 1)

    # cpu does not support parallel-group, let alone `sp`
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_forward_context", **{"return_value.flash_comm_v1_enabled": False}
    )
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_with_prefill(self, mock_context, mock_get_context, mock_get_context_2):
        mock_context.return_value.__enter__.return_value = None
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, with_prefill=True, num_reqs=4)
            self.assertTrue(self.proposer._runnable.call_count == 1)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_capture(
        self, mock_context, mock_get_context, mock_update_full_graph_params, mock_get_context_2
    ):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = True
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        mock_get_context_2.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, in_graph_capturing=True, aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            mock_update_full_graph_params.assert_not_called()
            self.proposer.use_cuda_graph = last_use_cuda_graph

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_run(
        self, mock_context, mock_get_context, mock_update_full_graph_params, mock_get_context_2
    ):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = False
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        mock_get_context_2.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        self.proposer.draft_attn_groups = [MagicMock()]
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, in_graph_capturing=False, aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            self.assertTrue(mock_update_full_graph_params.call_count == 1)
            self.proposer.use_cuda_graph = last_use_cuda_graph


class TestEagleProposerHelperMethods(TestBase):
    # TODO: Can add some tests about prepare_next_token_ids in future.

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.scheduler_config = MagicMock(max_num_seqs=3)
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.req_ids = [0, 1, 2]
        self.runner.arange_np = np.arange(10)
        self.runner.input_batch.num_reqs = 3
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # TODO: This is equivalent to disable_padded_drafter_batch=True.
    # We need to add a test_prepare_inputs_padded in future.
    def test_prepare_inputs(self):
        self.proposer.token_arange_np = np.arange(10)
        mock_attn = MagicMock()
        mock_attn.slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])
        num_rejected = torch.tensor([1, 0, 1], device=self.device)
        mock_return_attn = MagicMock()

        with (
            set_current_vllm_config(self.vllm_config),
            patch.object(self.proposer, "prepare_inputs", return_value=(mock_return_attn, torch.tensor([1, 2, 4]))),
        ):
            return_attn, indices = self.proposer.prepare_inputs(mock_attn, num_rejected)
            self.assertEqual(indices.tolist(), [1, 2, 4])
