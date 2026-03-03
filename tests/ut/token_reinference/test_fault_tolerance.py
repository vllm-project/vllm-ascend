import unittest
import json
import hashlib
import queue
from unittest.mock import MagicMock, patch, call, ANY
import torch
import numpy as np
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import logger
from tests.ut.base import TestBase
from vllm_ascend.token_reinference.fault_tolerance import FaultTolerance
from vllm_ascend.token_reinference.common import FaultAction, RecoveryStatus
from vllm_ascend.token_reinference.recovery_context import RecoveryContext
class TestFaultTolerance(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.model_runner = MagicMock()
        self.execute_model_func = MagicMock()

        patcher_dist_init = patch('torch.distributed.is_initialized', return_value=False)
        self.mock_dist_is_initialized = patcher_dist_init.start()
        self.addCleanup(patcher_dist_init.stop)

        patcher_dist_world = patch('torch.distributed.get_world_size', return_value=1)
        self.mock_dist_get_world_size = patcher_dist_world.start()
        self.addCleanup(patcher_dist_world.stop)

        patcher_dist_rank = patch('torch.distributed.get_rank', return_value=0)
        self.mock_dist_get_rank = patcher_dist_rank.start()
        self.addCleanup(patcher_dist_rank.stop)

        patcher_npu_restart = patch('torch_npu.npu.restart_device')
        self.mock_restart_device = patcher_npu_restart.start()
        self.addCleanup(patcher_npu_restart.stop)

        patcher_npu_reinit = patch('torch.distributed.reinit_process_group')
        self.mock_reinit_process_group = patcher_npu_reinit.start()
        self.addCleanup(patcher_npu_reinit.stop)

        patcher_npu_get_rng = patch('torch_npu.npu.get_rng_state', return_value=torch.tensor([1,2,3]))
        self.mock_get_rng_state = patcher_npu_get_rng.start()
        self.addCleanup(patcher_npu_get_rng.stop)

        patcher_npu_set_rng = patch('torch_npu.npu.set_rng_state')
        self.mock_set_rng_state = patcher_npu_set_rng.start()
        self.addCleanup(patcher_npu_set_rng.stop)

        patcher_npu_current_device = patch('torch_npu.npu.current_device', return_value=0)
        self.mock_current_device = patcher_npu_current_device.start()
        self.addCleanup(patcher_npu_current_device.stop)

        patcher_gc_collect = patch('gc.collect', return_value=0)
        self.mock_gc_collect = patcher_gc_collect.start()
        self.addCleanup(patcher_gc_collect.stop)

        patcher_npu_empty_cache = patch('torch_npu.npu.empty_cache', return_value=0)
        self.mock_empty_cache = patcher_npu_empty_cache.start()
        self.addCleanup(patcher_npu_empty_cache.stop)

        patcher_npu_sync = patch('torch.npu.synchronize')
        self.mock_npu_sync = patcher_npu_sync.start()
        self.addCleanup(patcher_npu_sync.stop)

        patcher_fault_aware = patch('vllm_ascend.token_reinference.fault_tolerance.FaultAware')
        mock_fault_aware_class = patcher_fault_aware.start()
        mock_fault_aware_instance = MagicMock()
        mock_fault_aware_class.return_value = mock_fault_aware_instance
        self.addCleanup(patcher_fault_aware.stop)

        self.logger_patcher = patch('vllm_ascend.token_reinference.fault_tolerance.logger')
        self.mock_logger = self.logger_patcher.start()
        self.addCleanup(self.logger_patcher.stop)

        self.ft = FaultTolerance(
            vllm_config=self.vllm_config,
            model_runner=self.model_runner,
            execute_model_func=self.execute_model_func,
        )
        self.ft.recovery_handler_manager = MagicMock()
        FaultTolerance._recovery_group = None
        FaultTolerance._sync_group = None

    def test_init_recovery_group_when_dist_not_initialized(self):
        with patch('torch.distributed.new_group') as mock_new_group:
            self.ft._init_recovery_group()
            mock_new_group.assert_not_called()

    def test_init_recovery_group_when_world_size_1(self):
        self.mock_dist_is_initialized.return_value = True
        self.mock_dist_get_world_size.return_value = 1
        with patch('torch.distributed.new_group') as mock_new_group:
            self.ft._init_recovery_group()
            mock_new_group.assert_not_called()

    def test_init_recovery_group_success(self):
        self.mock_dist_is_initialized.return_value = True
        self.mock_dist_get_world_size.return_value = 4
        self.ft.world_size = 4
        with patch('torch.distributed.new_group') as mock_new_group:
            self.ft._init_recovery_group()
            mock_new_group.assert_called_once_with(ranks=None, timeout=ANY, backend='gloo')
            self.mock_logger.info.assert_called()

    def test_init_sync_group_when_dist_not_initialized(self):
        self.mock_dist_is_initialized.return_value = False
        with patch('torch.distributed.new_group') as mock_new_group:
            self.ft._init_sync_group()
            mock_new_group.assert_not_called()

    def test_init_sync_group_success(self):
        self.mock_dist_is_initialized.return_value = True
        self.mock_dist_get_world_size.return_value = 4
        self.ft.world_size = 4
        with patch('torch.distributed.new_group') as mock_new_group:
            self.ft._init_sync_group()
            mock_new_group.assert_called_once_with(ranks=None, timeout=ANY, backend='hccl')

    @patch('vllm_ascend.token_reinference.fault_tolerance.RecoveryHandlerManager')
    @patch('vllm_ascend.token_reinference.fault_tolerance.ForceStopHandler')
    @patch('vllm_ascend.token_reinference.fault_tolerance.NetworkHandler')
    def test_build_recovery_handler_manager(self, mock_network, mock_force, mock_manager_cls):
        mock_manager = MagicMock()
        mock_manager_cls.return_value = mock_manager

        result = self.ft._build_recovery_handler_manager()

        mock_manager_cls.assert_called_once()
        mock_force.assert_called_once()
        mock_network.assert_called_once()
        expected_calls = [call(mock_force.return_value), call(mock_network.return_value)]
        mock_manager.register_handler.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(result, mock_manager)

    def test_handle_exception_no_handler(self):
        ctx = MagicMock(spec=RecoveryContext)
        self.ft.recovery_handler_manager.find_handler.return_value = None
        result = self.ft._handle_exception(ctx)
        self.assertTrue(torch.equal(result, FaultAction.RAISE_EXCEPTION))

    @patch.object(FaultTolerance, '_all_gather_for_recovery_group')
    @patch.object(FaultTolerance, '_clean_fault')
    @patch.object(FaultTolerance, '_coordinate_recovery')
    def test_handle_exception_with_handler_recover_success(self, mock_coord, mock_clean, mock_ag):
        mock_handler = MagicMock()
        self.ft.recovery_handler_manager.find_handler.return_value = mock_handler
        self.ft.stop_event = MagicMock()
        mock_clean.return_value = RecoveryStatus.SUCCESS
        mock_coord.side_effect = [FaultAction.RECOMPUTE, FaultAction.RECOMPUTE]
        ctx = MagicMock(spec=RecoveryContext)

        result = self.ft._handle_exception(ctx)

        self.assertTrue(torch.equal(result, FaultAction.RECOMPUTE))
        self.ft.stop_event.wait.assert_called_once()
        self.ft.stop_event.clear.assert_called_once()
        mock_ag.assert_called_once()
        mock_clean.assert_called_once_with(ctx)
        mock_handler.recover.assert_called_once_with(ctx)

    @patch.object(FaultTolerance, '_all_gather_for_recovery_group')
    @patch.object(FaultTolerance, '_clean_fault', return_value=RecoveryStatus.FAILED)
    @patch.object(FaultTolerance, '_coordinate_recovery', return_value=FaultAction.RAISE_EXCEPTION)
    def test_handle_exception_clean_failure(self, mock_coord, mock_clean, mock_ag):
        mock_handler = MagicMock()
        self.ft.recovery_handler_manager.find_handler.return_value = mock_handler
        self.ft.stop_event = MagicMock()
        ctx = MagicMock(spec=RecoveryContext)

        result = self.ft._handle_exception(ctx)

        self.assertTrue(torch.equal(result, FaultAction.RAISE_EXCEPTION))
        mock_clean.assert_called_once_with(ctx)
        mock_coord.assert_called_once_with(RecoveryStatus.FAILED)
        mock_handler.recover.assert_not_called()

    def test_coordinate_recovery_single_node_success(self):
        self.mock_dist_is_initialized.return_value = False
        result = self.ft._coordinate_recovery(RecoveryStatus.SUCCESS)
        self.assertTrue(torch.equal(result, FaultAction.RECOMPUTE))

    def test_coordinate_recovery_single_node_failure(self):
        self.mock_dist_is_initialized.return_value = False
        result = self.ft._coordinate_recovery(RecoveryStatus.FAILED)
        self.assertTrue(torch.equal(result, FaultAction.RAISE_EXCEPTION))

    def test_coordinate_recovery_multi_node_rank0(self):
        self.mock_dist_is_initialized.return_value = True
        self.mock_dist_get_world_size.return_value = 2
        self.ft.world_size = 2
        self.ft.rank = 0

        local_status = RecoveryStatus.SUCCESS
        all_statuses = [RecoveryStatus.SUCCESS, RecoveryStatus.FAILED]
        decisions = [FaultAction.RETURN, FaultAction.RAISE_EXCEPTION]

        with patch.object(self.ft, '_gather_statuses', return_value=all_statuses) as mock_gather, \
                patch.object(self.ft, '_analyze_global_status', return_value=decisions) as mock_analyze, \
                patch.object(self.ft, '_scatter_ft_actions', return_value=FaultAction.RETURN) as mock_scatter:
            result = self.ft._coordinate_recovery(local_status)

        mock_gather.assert_called_once_with(local_status)
        mock_analyze.assert_called_once_with(all_statuses)
        mock_scatter.assert_called_once_with(decisions)
        self.assertTrue(torch.equal(result, FaultAction.RETURN))

    def test_coordinate_recovery_multi_node_non_rank0(self):
        self.mock_dist_is_initialized.return_value = True
        self.mock_dist_get_world_size.return_value = 2
        self.ft.world_size = 2
        self.ft.rank = 1

        local_status = RecoveryStatus.SUCCESS

        with patch.object(self.ft, '_gather_statuses', return_value=[]) as mock_gather, \
                patch.object(self.ft, '_analyze_global_status') as mock_analyze, \
                patch.object(self.ft, '_receive_ft_actions', return_value=FaultAction.RECOMPUTE) as mock_receive:
            result = self.ft._coordinate_recovery(local_status)

        mock_gather.assert_called_once_with(local_status)
        mock_analyze.assert_not_called()
        mock_receive.assert_called_once()
        self.assertTrue(torch.equal(result, FaultAction.RECOMPUTE))

    def test_clean_fault_queue(self):
        q:queue.Queue = queue.Queue()
        q.put(1)
        q.put(2)
        self.ft.fault_queue = q
        self.ft._clean_fault_queue()
        self.assertTrue(q.empty())

    @patch.object(FaultTolerance, '_clean_fault_queue')
    @patch.object(FaultTolerance, '_restore_essential_state')
    def test_clean_fault_success_dummy_run_true(self, mock_restore, mock_clean_q):
        ctx = MagicMock()
        ctx.is_dummy_run = True
        ctx.back_up = {'some': 'data'}

        status = self.ft._clean_fault(ctx)

        self.assertTrue(torch.equal(status, RecoveryStatus.SUCCESS))
        mock_clean_q.assert_called_once()
        self.mock_restart_device.assert_called_once_with(0)
        self.mock_reinit_process_group.assert_called_once_with(group=None, rebuild_link=False)
        mock_restore.assert_called_once_with(ctx.back_up)

    @patch.object(FaultTolerance, '_clean_fault_queue')
    @patch.object(FaultTolerance, '_restore_essential_state')
    def test_clean_fault_success_dummy_run_false(self, mock_restore, mock_clean_q):
        ctx = MagicMock()
        ctx.is_dummy_run = False

        status = self.ft._clean_fault(ctx)

        self.assertTrue(torch.equal(status, RecoveryStatus.SUCCESS))
        mock_clean_q.assert_called_once()
        self.mock_restart_device.assert_called_once()
        self.mock_reinit_process_group.assert_called_once()
        mock_restore.assert_not_called()

    def test_clean_fault_exception(self):
        self.mock_restart_device.side_effect = Exception("device restart failed")
        ctx = MagicMock()
        status = self.ft._clean_fault(ctx)
        self.assertTrue(torch.equal(status, RecoveryStatus.FAILED))
        self.mock_logger.error.assert_called_once()

    def test_all_gather_for_recovery_group(self):
        FaultTolerance._recovery_group = MagicMock()
        self.ft.world_size = 2
        with patch('torch.distributed.all_gather') as mock_all_gather:
            self.ft._all_gather_for_recovery_group()
            mock_all_gather.assert_called_once()
            args, kwargs = mock_all_gather.call_args
            self.assertEqual(len(args[0]), 2)
            self.assertEqual(kwargs['group'], FaultTolerance._recovery_group)

    def test_all_gather_for_recovery_group_exception(self):
        FaultTolerance._recovery_group = MagicMock()
        with patch('torch.distributed.all_gather', side_effect=Exception("gather failed")) as mock_all_gather:
            with self.assertRaises(Exception) as ctx:
                self.ft._all_gather_for_recovery_group()
            self.assertEqual(str(ctx.exception), "gather failed")
            self.mock_logger.error.assert_called_once()

    def test_all_gather_for_sync_group(self):
        FaultTolerance._sync_group = MagicMock()
        self.ft.world_size = 2
        with patch('torch.distributed.all_gather') as mock_all_gather:
            with patch('torch.tensor', return_value=torch.tensor(0)):
                self.ft._all_gather_for_sync_group()
            mock_all_gather.assert_called_once()
            args, kwargs = mock_all_gather.call_args
            self.assertEqual(len(args[0]), 2)
            self.assertEqual(kwargs['group'], FaultTolerance._sync_group)
            self.mock_npu_sync.assert_called_once()

    def test_gather_statuses_rank0(self):
        self.ft.rank = 0
        self.ft.world_size = 2
        local_status = torch.tensor([RecoveryStatus.SUCCESS])
        FaultTolerance._recovery_group = MagicMock()

        with patch('torch.distributed.gather') as mock_gather:
            result = self.ft._gather_statuses(local_status)

            mock_gather.assert_called_once_with(
                local_status,
                gather_list=ANY,
                dst=0,
                group=FaultTolerance._recovery_group
            )
            self.assertEqual(len(result), 2)

    def test_gather_statuses_non_rank0(self):
        self.ft.rank = 1
        self.ft.world_size = 2
        local_status = torch.tensor([RecoveryStatus.SUCCESS])
        FaultTolerance._recovery_group = MagicMock()

        with patch('torch.distributed.gather') as mock_gather:
            result = self.ft._gather_statuses(local_status)

            mock_gather.assert_called_once_with(
                local_status,
                gather_list=None,
                dst=0,
                group=FaultTolerance._recovery_group
            )
            self.assertEqual(result, [])

    def test_gather_statuses_exception_rank0(self):
        self.ft.rank = 0
        self.ft.world_size = 2
        local_status = torch.tensor([RecoveryStatus.SUCCESS])
        FaultTolerance._recovery_group = MagicMock()

        with patch('torch.distributed.gather', side_effect=Exception("gather error")) as mock_gather:
            result = self.ft._gather_statuses(local_status)

            self.assertEqual(len(result), 2)
            for r in result:
                self.assertTrue(torch.equal(r, RecoveryStatus.FAILED))

    def test_analyze_global_status_all_success(self):
        self.ft.world_size = 2
        all_status = [RecoveryStatus.SUCCESS, RecoveryStatus.SUCCESS]
        decisions = self.ft._analyze_global_status(all_status)
        self.assertEqual(len(decisions), 2)
        for d in decisions:
            self.assertTrue(torch.equal(d, FaultAction.RECOMPUTE))

    def test_analyze_global_status_all_failure(self):
        self.ft.world_size = 2
        all_status = [RecoveryStatus.FAILED, RecoveryStatus.FAILED]
        decisions = self.ft._analyze_global_status(all_status)
        self.assertEqual(len(decisions), 2)
        for d in decisions:
            self.assertTrue(torch.equal(d, FaultAction.RAISE_EXCEPTION))

    def test_analyze_global_status_mixed(self):
        self.ft.world_size = 2
        all_status = [RecoveryStatus.SUCCESS, RecoveryStatus.FAILED]
        decisions = self.ft._analyze_global_status(all_status)
        self.assertEqual(len(decisions), 2)
        self.assertTrue(torch.equal(decisions[0], FaultAction.RETURN))
        self.assertTrue(torch.equal(decisions[1], FaultAction.RAISE_EXCEPTION))

    def test_analyze_global_status_unknown(self):
        self.ft.world_size = 2
        unknown = torch.tensor([999])
        all_status = [RecoveryStatus.SUCCESS, unknown]
        with patch.object(self.mock_logger, 'warning') as mock_warn:
            decisions = self.ft._analyze_global_status(all_status)
        self.assertEqual(len(decisions), 2)
        self.assertTrue(torch.equal(decisions[0], FaultAction.RETURN))
        self.assertTrue(torch.equal(decisions[1], FaultAction.RAISE_EXCEPTION))
        self.assertEqual(mock_warn.call_count, 2)

    def test_scatter_ft_actions(self):
        ft_actions = [torch.zeros_like(FaultAction.RECOMPUTE), torch.zeros_like(FaultAction.RETURN)]
        FaultTolerance._recovery_group = MagicMock()
        with patch('torch.distributed.scatter') as mock_scatter:
            def scatter_side_effect(recv_tensor, *args, **kwargs):
                recv_tensor.copy_(torch.zeros_like(FaultAction.RECOMPUTE))
            mock_scatter.side_effect = scatter_side_effect
            result = self.ft._scatter_ft_actions(ft_actions)
            mock_scatter.assert_called_once_with(
                ANY,
                scatter_list=ft_actions,
                src=0,
                group=FaultTolerance._recovery_group
            )
            self.assertTrue(torch.equal(result, torch.zeros_like(FaultAction.RECOMPUTE)))

    def test_receive_ft_actions(self):
        FaultTolerance._recovery_group = MagicMock()
        with patch('torch.distributed.scatter') as mock_scatter:
            def scatter_side_effect(recv_tensor, *args, **kwargs):
                recv_tensor.copy_(torch.zeros_like(FaultAction.RETURN))

            mock_scatter.side_effect = scatter_side_effect

            result = self.ft._receive_ft_actions()

            mock_scatter.assert_called_once_with(
                ANY,
                scatter_list=None,
                src=0,
                group=FaultTolerance._recovery_group
            )
            self.assertTrue(torch.equal(result, torch.zeros_like(FaultAction.RETURN)))

    def test_generate_scheduler_output_key_normal(self):
        mock_out = MagicMock()
        req1 = MagicMock()
        req1.req_id = 'a'
        req2 = MagicMock()
        req2.req_id = 'b'
        mock_out.scheduled_new_reqs = [req1, req2]
        mock_cached = MagicMock()
        mock_cached.req_ids = ['c1', 'c2']
        mock_cached.num_computed_tokens = [5, 10]
        mock_out.scheduled_cached_reqs = mock_cached

        expected_data = {
            'new_req_ids': ['a', 'b'],
            'cached_req_ids': ['c1', 'c2'],
            'cached_num_tokens': [5, 10]
        }
        expected_json = json.dumps(expected_data, sort_keys=True, ensure_ascii=False)
        expected_hash = hashlib.sha256(expected_json.encode('utf-8')).hexdigest()

        result = self.ft._generate_scheduler_output_key(mock_out)
        self.assertEqual(result, expected_hash)

    def test_generate_scheduler_output_key_empty(self):
        mock_out = MagicMock()
        mock_out.scheduled_new_reqs = []
        mock_cached = MagicMock()
        mock_cached.req_ids = []
        mock_cached.num_computed_tokens = []
        mock_out.scheduled_cached_reqs = mock_cached

        expected_data: dict[str,Any] = {
            'new_req_ids': [],
            'cached_req_ids': [],
            'cached_num_tokens': []
        }
        expected_json = json.dumps(expected_data, sort_keys=True, ensure_ascii=False)
        expected_hash = hashlib.sha256(expected_json.encode('utf-8')).hexdigest()

        result = self.ft._generate_scheduler_output_key(mock_out)
        self.assertEqual(result, expected_hash)

    def test_create_essential_state_backup_full(self):
        import numpy as np

        self.model_runner.requests = {
            'req1': MagicMock(
                output_token_ids=[1, 2, 3],
                num_computed_tokens=5,
                block_ids=(MagicMock(), MagicMock())
            )
        }

        self.model_runner.input_batch = MagicMock()
        ib = self.model_runner.input_batch
        ib._req_ids = ['r1']
        ib.req_output_token_ids = [[1], [2]]
        ib.req_id_to_index = {'r1': 0}
        ib.spec_token_ids = [[3]]
        ib.block_table.block_tables = [MagicMock(num_blocks_per_row=[4, 5])]
        ib.batch_update_builder._removed = [6]
        ib.batch_update_builder.added = [7]

        ib.token_ids_cpu = np.array([8], dtype=np.int32)
        ib.num_tokens = np.array([9], dtype=np.int32)
        ib.num_tokens_no_spec = np.array([10], dtype=np.int32)
        ib.num_computed_tokens_cpu = np.array([11], dtype=np.int32)
        ib.num_accepted_tokens_cpu = np.array([12], dtype=np.int32)
        ib.prev_sampled_token_ids = torch.tensor([13])

        self.model_runner.eplb_updator = MagicMock(
            update_info_all={'info': 'test'},
            reqs=['r1'],
            cur_iterations=2
        )

        backup = self.ft._create_essential_state_backup('arg', kw='val')

        self.assertEqual(backup['args'], ('arg',))
        self.assertEqual(backup['kwargs'], {'kw': 'val'})

        self.assertTrue(torch.equal(backup['generator_state'], torch.tensor([1, 2, 3])))
        self.mock_get_rng_state.assert_called_once()

        req_backup = backup['requests_essential']['req1']
        self.assertEqual(req_backup['output_token_ids'], [1, 2, 3])
        self.assertEqual(req_backup['num_computed_tokens'], 5)
        self.assertEqual(len(req_backup['block_ids']), 2)

        self.assertEqual(backup['_req_ids'], ['r1'])
        self.assertEqual(backup['req_output_token_ids'], [[1], [2]])
        self.assertEqual(backup['req_id_to_index'], {'r1': 0})
        self.assertEqual(backup['spec_token_ids'], [[3]])
        self.assertEqual(backup['num_blocks_per_row'], [[4, 5]])
        self.assertEqual(backup['_removed'], [6])
        self.assertEqual(backup['added'], [7])
        self.assertEqual(backup['token_ids_cpu'].tolist(), [8])
        self.assertEqual(backup['num_tokens'].tolist(), [9])
        self.assertEqual(backup['num_tokens_no_spec'].tolist(), [10])
        self.assertEqual(backup['num_computed_tokens_cpu'].tolist(), [11])
        self.assertEqual(backup['num_accepted_tokens_cpu'].tolist(), [12])
        self.assertTrue(torch.equal(backup['prev_sampled_token_ids'], torch.tensor([13])))

        self.assertEqual(backup['update_info_all'], {'info': 'test'})
        self.assertEqual(backup['reqs'], ['r1'])
        self.assertEqual(backup['cur_iterations'], 2)

    def test_restore_essential_state_full(self):
        import numpy as np
        from types import SimpleNamespace

        backup = {
            'args': ('arg',),
            'kwargs': {'kw': 'val'},
            'generator_state': torch.tensor([99]),
            'requests_essential': {
                'req1': {
                    'output_token_ids': [4, 5, 6],
                    'num_computed_tokens': 7,
                    'block_ids': (MagicMock(), MagicMock())
                }
            },
            '_req_ids': ['r2'],
            'req_output_token_ids': [[8]],
            'req_id_to_index': {'r2': 1},
            'spec_token_ids': [[9]],
            'num_blocks_per_row': [[10, 11]],
            '_removed': [12],
            'added': [13],
            'token_ids_cpu': np.array([14], dtype=np.int32),
            'num_tokens': np.array([15], dtype=np.int32),
            'num_tokens_no_spec': np.array([16], dtype=np.int32),
            'num_computed_tokens_cpu': np.array([17], dtype=np.int32),
            'num_accepted_tokens_cpu': np.array([18], dtype=np.int32),
            'prev_sampled_token_ids': torch.tensor([19]),
            'update_info_all': {'info2': 'test2'},
            'reqs': ['r2'],
            'cur_iterations': 3
        }

        self.model_runner.requests = {'req1': MagicMock()}
        self.model_runner.input_batch = SimpleNamespace()
        ib = self.model_runner.input_batch

        ib._req_ids = []
        ib.req_output_token_ids = []
        ib.req_id_to_index = {}
        ib.spec_token_ids = []
        ib.block_table = SimpleNamespace()
        ib.block_table.block_tables = [SimpleNamespace()]
        ib.block_table.block_tables[0].num_blocks_per_row = []
        ib.batch_update_builder = SimpleNamespace()
        ib.batch_update_builder._removed = []
        ib.batch_update_builder.added = []
        ib.token_ids_cpu = np.array([0], dtype=np.int32)
        ib.num_tokens = np.array([0], dtype=np.int32)
        ib.num_tokens_no_spec = np.array([0], dtype=np.int32)
        ib.num_computed_tokens_cpu = np.array([0], dtype=np.int32)
        ib.num_accepted_tokens_cpu = np.array([0], dtype=np.int32)
        ib.prev_sampled_token_ids = torch.tensor([0])

        self.model_runner.eplb_updator = MagicMock()

        self.ft._restore_essential_state(backup)

        self.mock_set_rng_state.assert_called_once_with(torch.tensor([99]))

        req_state = self.model_runner.requests['req1']
        self.assertEqual(req_state.output_token_ids, [4, 5, 6])
        self.assertEqual(req_state.num_computed_tokens, 7)
        self.assertEqual(req_state.block_ids, backup['requests_essential']['req1']['block_ids'])

        self.assertEqual(ib._req_ids, ['r2'])
        self.assertEqual(ib.req_output_token_ids, [[8]])
        self.assertEqual(ib.req_id_to_index, {'r2': 1})
        self.assertEqual(ib.spec_token_ids, [[9]])
        self.assertEqual(ib.block_table.block_tables[0].num_blocks_per_row, [10, 11])
        self.assertEqual(ib.batch_update_builder._removed, [12])
        self.assertEqual(ib.batch_update_builder.added, [13])

        np.testing.assert_array_equal(ib.token_ids_cpu, np.array([14]))
        np.testing.assert_array_equal(ib.num_tokens, np.array([15]))
        np.testing.assert_array_equal(ib.num_tokens_no_spec, np.array([16]))
        np.testing.assert_array_equal(ib.num_computed_tokens_cpu, np.array([17]))
        np.testing.assert_array_equal(ib.num_accepted_tokens_cpu, np.array([18]))
        self.assertTrue(torch.equal(ib.prev_sampled_token_ids, torch.tensor([19])))

        eplb = self.model_runner.eplb_updator
        self.assertEqual(eplb.update_info_all, {'info2': 'test2'})
        self.assertEqual(eplb.reqs, ['r2'])
        self.assertEqual(eplb.cur_iterations, 3)

        self.mock_empty_cache.assert_called_once()
        self.mock_gc_collect.assert_called_once()

    def test_restore_essential_state_empty_backup(self):
        self.ft._restore_essential_state({})
        self.mock_set_rng_state.assert_not_called()

    def test_single_node_decision(self):
        self.assertTrue(torch.equal(
            self.ft._single_node_decision(RecoveryStatus.SUCCESS),
            FaultAction.RECOMPUTE
        ))
        self.assertTrue(torch.equal(
            self.ft._single_node_decision(RecoveryStatus.FAILED),
            FaultAction.RAISE_EXCEPTION
        ))

    if __name__ == '__main__':
        unittest.main()
