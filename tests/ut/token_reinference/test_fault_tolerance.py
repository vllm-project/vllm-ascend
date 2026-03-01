import unittest
import json
import hashlib
import queue
import threading
from unittest.mock import MagicMock, patch, call, ANY

import torch
import numpy as np

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from tests.ut.base import TestBase

from vllm_ascend.token_reinference.fault_tolerance import FaultTolerance
from vllm_ascend.token_reinference.common import FaultAction, RecoveryStatus
from vllm_ascend.token_reinference.recovery_context import RecoveryContext


class TestFaultTolerance(TestBase):
    """FaultTolerance 类的完整单元测试，风格与 test_worker 一致。"""

    def setUp(self):
        """每个测试用例前的初始化。"""
        super().setUp()

        # 创建基本模拟对象
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.model_runner = MagicMock()
        self.execute_model_func = MagicMock()

        # 模拟 torch.distributed 相关函数
        self.dist_patcher = patch.multiple(
            'torch.distributed',
            is_initialized=MagicMock(return_value=False),
            get_world_size=MagicMock(return_value=1),
            get_rank=MagicMock(return_value=0),
            new_group=MagicMock(),
            all_gather=MagicMock(),
            gather=MagicMock(),
            scatter=MagicMock(),
            reinit_process_group=MagicMock()
        )
        self.dist_mocks = self.dist_patcher.start()
        self.addCleanup(self.dist_patcher.stop)

        # 模拟 torch_npu 相关函数
        self.npu_patcher = patch.multiple(
            'torch_npu.npu',
            restart_device=MagicMock(),
            get_rng_state=MagicMock(return_value=torch.tensor([1, 2, 3])),
            set_rng_state=MagicMock(),
            current_device=MagicMock(return_value=0),
            synchronize=MagicMock()
        )
        self.npu_mocks = self.npu_patcher.start()
        self.addCleanup(self.npu_patcher.stop)

        # 模拟 FaultAware 避免启动真实线程
        self.fault_aware_patcher = patch(
            'vllm_ascend.token_reinference.fault_aware.FaultAware'
        )
        self.mock_fault_aware_class = self.fault_aware_patcher.start()
        self.mock_fault_aware_instance = MagicMock()
        self.mock_fault_aware_class.return_value = self.mock_fault_aware_instance
        self.addCleanup(self.fault_aware_patcher.stop)

        # 模拟 logger
        self.logger_patcher = patch('vllm_ascend.token_reinference.fault_tolerance.logger')
        self.mock_logger = self.logger_patcher.start()
        self.addCleanup(self.logger_patcher.stop)

        # 创建 FaultTolerance 实例
        self.ft = FaultTolerance(
            vllm_config=self.vllm_config,
            model_runner=self.model_runner,
            execute_model_func=self.execute_model_func
        )

        # 重置类变量（因为 new_group 会在 __init__ 中被调用，所以需要模拟）
        FaultTolerance._recovery_group = None
        FaultTolerance._sync_group = None

    def test_init_recovery_group_when_dist_not_initialized(self):
        """分布式未初始化时，不应创建组。"""
        self.dist_mocks['is_initialized'].return_value = False
        self.ft._init_recovery_group()
        self.dist_mocks['new_group'].assert_not_called()

    def test_init_recovery_group_when_world_size_1(self):
        """world_size == 1 时，不应创建组。"""
        self.dist_mocks['is_initialized'].return_value = True
        self.dist_mocks['get_world_size'].return_value = 1
        self.ft._init_recovery_group()
        self.dist_mocks['new_group'].assert_not_called()

    def test_init_recovery_group_success(self):
        """正常初始化 recovery 组。"""
        self.dist_mocks['is_initialized'].return_value = True
        self.dist_mocks['get_world_size'].return_value = 4
        self.dist_mocks['get_rank'].return_value = 2
        self.ft._init_recovery_group()
        self.dist_mocks['new_group'].assert_called_once_with(
            ranks=None, timeout=ANY, backend='gloo'
        )
        self.mock_logger.info.assert_called_once()

    def test_init_sync_group_when_dist_not_initialized(self):
        """分布式未初始化时，不应创建 sync 组。"""
        self.dist_mocks['is_initialized'].return_value = False
        self.ft._init_sync_group()
        self.dist_mocks['new_group'].assert_not_called()

    def test_init_sync_group_success(self):
        """正常初始化 sync 组。"""
        self.dist_mocks['is_initialized'].return_value = True
        self.dist_mocks['get_world_size'].return_value = 4
        self.ft._init_sync_group()
        self.dist_mocks['new_group'].assert_called_once_with(
            ranks=None, timeout=ANY, backend='hccl'
        )

    @patch('vllm_ascend.token_reinference.fault_tolerance.RecoveryHandlerManager')
    @patch('vllm_ascend.token_reinference.fault_tolerance.ForceStopHandler')
    @patch('vllm_ascend.token_reinference.fault_tolerance.NetworkHandler')
    def test_build_recovery_handler_manager(self, mock_network, mock_force, mock_manager_cls):
        """验证处理器管理器正确构建并注册了两个处理器。"""
        mock_manager = MagicMock()
        mock_manager_cls.return_value = mock_manager

        result = self.ft._build_recovery_handler_manager()

        mock_manager_cls.assert_called_once()
        mock_force.assert_called_once()
        mock_network.assert_called_once()
        expected_calls = [
            call(mock_force.return_value),
            call(mock_network.return_value)
        ]
        mock_manager.register_handler.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(result, mock_manager)

    def test_handle_exception_no_handler(self):
        """没有找到处理器时返回 RAISE_EXCEPTION。"""
        ctx = MagicMock(spec=RecoveryContext)
        self.ft.recovery_handler_manager.find_handler.return_value = None
        result = self.ft._handle_exception(ctx)
        self.assertTrue(torch.equal(result, FaultAction.RAISE_EXCEPTION))

    def test_handle_exception_with_handler_clean_success_recover_success(self):
        """处理器存在，清理成功，恢复成功 → 返回 RECOMPUTE。"""
        ctx = MagicMock(spec=RecoveryContext)
        mock_handler = MagicMock()
        self.ft.recovery_handler_manager.find_handler.return_value = mock_handler
        self.ft.stop_event = MagicMock()
        with patch.object(self.ft, '_all_gather_for_recovery_group') as mock_ag, \
             patch.object(self.ft, '_clean_fault', return_value=RecoveryStatus.SUCCESS) as mock_clean, \
             patch.object(self.ft, '_coordinate_recovery') as mock_coord:
            mock_coord.side_effect = [FaultAction.RECOMPUTE, RecoveryStatus.SUCCESS]
            result = self.ft._handle_exception(ctx)

        self.assertTrue(torch.equal(result, FaultAction.RECOMPUTE))
        self.ft.stop_event.wait.assert_called_once()
        self.ft.stop_event.clear.assert_called_once()
        mock_ag.assert_called_once()
        mock_clean.assert_called_once_with(ctx)
        mock_handler.recover.assert_called_once_with(ctx)

    def test_handle_exception_clean_failure(self):
        """清理失败，协调后可能返回非 RECOMPUTE。"""
        ctx = MagicMock(spec=RecoveryContext)
        mock_handler = MagicMock()
        self.ft.recovery_handler_manager.find_handler.return_value = mock_handler
        self.ft.stop_event = MagicMock()
        with patch.object(self.ft, '_all_gather_for_recovery_group'), \
             patch.object(self.ft, '_clean_fault', return_value=RecoveryStatus.FAILED) as mock_clean, \
             patch.object(self.ft, '_coordinate_recovery', return_value=FaultAction.RAISE_EXCEPTION) as mock_coord:
            result = self.ft._handle_exception(ctx)

        self.assertTrue(torch.equal(result, FaultAction.RAISE_EXCEPTION))
        mock_clean.assert_called_once()
        mock_coord.assert_called_once_with(RecoveryStatus.FAILED)
        mock_handler.recover.assert_not_called()


    def test_coordinate_recovery_single_node_success(self):
        """单节点，本地状态 SUCCESS → RECOMPUTE。"""
        self.dist_mocks['is_initialized'].return_value = False
        result = self.ft._coordinate_recovery(RecoveryStatus.SUCCESS)
        self.assertTrue(torch.equal(result, FaultAction.RECOMPUTE))

    def test_coordinate_recovery_single_node_failure(self):
        """单节点，本地状态 FAILED → RAISE_EXCEPTION。"""
        self.dist_mocks['is_initialized'].return_value = False
        result = self.ft._coordinate_recovery(RecoveryStatus.FAILED)
        self.assertTrue(torch.equal(result, FaultAction.RAISE_EXCEPTION))

    def test_coordinate_recovery_multi_node_rank0(self):
        """多节点，rank0 收集状态并分发。"""
        self.dist_mocks['is_initialized'].return_value = True
        self.dist_mocks['get_world_size'].return_value = 2
        self.ft.world_size = 2
        self.ft.rank = 0

        local_status = RecoveryStatus.SUCCESS
        all_statuses = [RecoveryStatus.SUCCESS, RecoveryStatus.FAILED]
        decisions = [FaultAction.RECOMPUTE, FaultAction.RETURN]

        with patch.object(self.ft, '_gather_statuses', return_value=all_statuses) as mock_gather, \
             patch.object(self.ft, '_analyze_global_status', return_value=decisions) as mock_analyze, \
             patch.object(self.ft, '_scatter_ft_actions', return_value=FaultAction.RECOMPUTE) as mock_scatter:
            result = self.ft._coordinate_recovery(local_status)

        mock_gather.assert_called_once_with(local_status)
        mock_analyze.assert_called_once_with(all_statuses)
        mock_scatter.assert_called_once_with(decisions)
        self.assertTrue(torch.equal(result, FaultAction.RECOMPUTE))

    def test_coordinate_recovery_multi_node_non_rank0(self):
        """多节点，非 rank0 接收动作。"""
        self.dist_mocks['is_initialized'].return_value = True
        self.dist_mocks['get_world_size'].return_value = 2
        self.ft.world_size = 2
        self.ft.rank = 1

        local_status = RecoveryStatus.SUCCESS
        with patch.object(self.ft, '_gather_statuses', return_value=[]) as mock_gather, \
             patch.object(self.ft, '_analyze_global_status') as mock_analyze, \
             patch.object(self.ft, '_receive_ft_actions', return_value=FaultAction.RETURN) as mock_receive:
            result = self.ft._coordinate_recovery(local_status)

        mock_gather.assert_called_once_with(local_status)
        mock_analyze.assert_not_called()
        mock_receive.assert_called_once()
        self.assertTrue(torch.equal(result, FaultAction.RETURN))


    def test_clean_fault_queue(self):
        """验证队列被清空。"""
        q = queue.Queue()
        q.put(1)
        q.put(2)
        self.ft.fault_queue = q
        self.ft._clean_fault_queue()
        self.assertTrue(q.empty())

    def test_clean_fault_success_dummy_run_true(self):
        """清理成功，dummy_run=True 时恢复状态。"""
        ctx = MagicMock()
        ctx.is_dummy_run = True
        ctx.back_up = {'some': 'data'}
        with patch.object(self.ft, '_clean_fault_queue') as mock_clean_q, \
             patch.object(self.ft, '_restore_essential_state') as mock_restore:
            status = self.ft._clean_fault(ctx)

        self.assertTrue(torch.equal(status, RecoveryStatus.SUCCESS))
        mock_clean_q.assert_called_once()
        self.npu_mocks['restart_device'].assert_called_once_with(0)
        self.dist_mocks['reinit_process_group'].assert_called_once_with(group=None, rebuild_link=False)
        mock_restore.assert_called_once_with(ctx.back_up)

    def test_clean_fault_success_dummy_run_false(self):
        """清理成功，dummy_run=False 时不恢复状态。"""
        ctx = MagicMock()
        ctx.is_dummy_run = False
        with patch.object(self.ft, '_clean_fault_queue'), \
             patch.object(self.ft, '_restore_essential_state') as mock_restore:
            status = self.ft._clean_fault(ctx)

        self.assertTrue(torch.equal(status, RecoveryStatus.SUCCESS))
        mock_restore.assert_not_called()

    def test_clean_fault_exception(self):
        """清理过程中发生异常，返回 FAILED。"""
        ctx = MagicMock()
        self.npu_mocks['restart_device'].side_effect = Exception("device restart failed")
        status = self.ft._clean_fault(ctx)
        self.assertTrue(torch.equal(status, RecoveryStatus.FAILED))
        self.mock_logger.error.assert_called_once()

    def test_all_gather_for_recovery_group(self):
        """验证 recovery 组的 all_gather 被调用。"""
        self.ft.world_size = 2
        self.dist_mocks['all_gather'].return_value = None
        self.ft._all_gather_for_recovery_group()
        self.dist_mocks['all_gather'].assert_called_once()
        args, kwargs = self.dist_mocks['all_gather'].call_args
        self.assertEqual(len(args[1]), 2)  # gather_list 长度为 world_size
        self.assertEqual(kwargs['group'], FaultTolerance._recovery_group)

    def test_all_gather_for_recovery_group_exception(self):
        """all_gather 抛出异常，应继续抛出。"""
        self.dist_mocks['all_gather'].side_effect = Exception("gather failed")
        with self.assertRaises(Exception) as ctx:
            self.ft._all_gather_for_recovery_group()
        self.assertEqual(str(ctx.exception), "gather failed")
        self.mock_logger.error.assert_called_once()

    def test_all_gather_for_sync_group(self):
        """验证 sync 组的 all_gather 被调用。"""
        self.ft.world_size = 2
        self.dist_mocks['all_gather'].return_value = None
        self.ft._all_gather_for_sync_group()
        self.dist_mocks['all_gather'].assert_called_once()
        args, kwargs = self.dist_mocks['all_gather'].call_args
        self.assertEqual(len(args[1]), 2)
        self.assertEqual(kwargs['group'], FaultTolerance._sync_group)
        self.npu_mocks['synchronize'].assert_called_once()

    def test_gather_statuses_rank0(self):
        """rank0 收集状态。"""
        self.ft.rank = 0
        self.ft.world_size = 2
        local_status = torch.tensor([RecoveryStatus.SUCCESS])
        expected_gather_list = [torch.zeros_like(local_status) for _ in range(2)]
        self.dist_mocks['gather'].return_value = None

        result = self.ft._gather_statuses(local_status)

        self.dist_mocks['gather'].assert_called_once_with(
            local_status,
            gather_list=ANY,
            dst=0,
            group=FaultTolerance._recovery_group
        )
        self.assertEqual(len(result), 2)

    def test_gather_statuses_non_rank0(self):
        """非 rank0 只发送不接收。"""
        self.ft.rank = 1
        self.ft.world_size = 2
        local_status = torch.tensor([RecoveryStatus.SUCCESS])

        result = self.ft._gather_statuses(local_status)

        self.dist_mocks['gather'].assert_called_once_with(
            local_status,
            gather_list=None,
            dst=0,
            group=FaultTolerance._recovery_group
        )
        self.assertEqual(result, [])

    def test_gather_statuses_exception_rank0(self):
        """rank0 异常时返回失败列表。"""
        self.ft.rank = 0
        self.ft.world_size = 2
        local_status = torch.tensor([RecoveryStatus.SUCCESS])
        self.dist_mocks['gather'].side_effect = Exception("gather error")

        result = self.ft._gather_statuses(local_status)

        self.assertEqual(len(result), 2)
        for r in result:
            self.assertTrue(torch.equal(r, RecoveryStatus.FAILED))

    def test_analyze_global_status_all_success(self):
        """所有状态为 SUCCESS → 全 RECOMPUTE。"""
        all_status = [RecoveryStatus.SUCCESS, RecoveryStatus.SUCCESS]
        decisions = self.ft._analyze_global_status(all_status)
        self.assertEqual(len(decisions), 2)
        for d in decisions:
            self.assertTrue(torch.equal(d, FaultAction.RECOMPUTE))

    def test_analyze_global_status_all_failure(self):
        """所有状态为 FAILED → 全 RAISE_EXCEPTION。"""
        all_status = [RecoveryStatus.FAILED, RecoveryStatus.FAILED]
        decisions = self.ft._analyze_global_status(all_status)
        self.assertEqual(len(decisions), 2)
        for d in decisions:
            self.assertTrue(torch.equal(d, FaultAction.RAISE_EXCEPTION))

    def test_analyze_global_status_mixed(self):
        """混合状态：成功者 RETURN，失败者 RAISE。"""
        all_status = [RecoveryStatus.SUCCESS, RecoveryStatus.FAILED]
        decisions = self.ft._analyze_global_status(all_status)
        self.assertEqual(len(decisions), 2)
        self.assertTrue(torch.equal(decisions[0], FaultAction.RETURN))
        self.assertTrue(torch.equal(decisions[1], FaultAction.RAISE_EXCEPTION))

    def test_analyze_global_status_unknown(self):
        """未知状态视为失败。"""
        unknown = torch.tensor([999])
        all_status = [RecoveryStatus.SUCCESS, unknown]
        with patch.object(self.mock_logger, 'warning') as mock_warn:
            decisions = self.ft._analyze_global_status(all_status)
        self.assertEqual(len(decisions), 2)
        self.assertTrue(torch.equal(decisions[0], FaultAction.RETURN))
        self.assertTrue(torch.equal(decisions[1], FaultAction.RAISE_EXCEPTION))
        mock_warn.assert_called_once()

    def test_scatter_ft_actions(self):
        """rank0 分发动作。"""
        ft_actions = [torch.tensor([FaultAction.RECOMPUTE]), torch.tensor([FaultAction.RETURN])]
        self.dist_mocks['scatter'].return_value = None
        result = self.ft._scatter_ft_actions(ft_actions)
        self.dist_mocks['scatter'].assert_called_once_with(
            ANY,
            scatter_list=ft_actions,
            src=0,
            group=FaultTolerance._recovery_group
        )
        self.assertTrue(torch.equal(result, torch.tensor([0])))  # 初始值

    def test_receive_ft_actions(self):
        """非 rank0 接收动作。"""
        self.dist_mocks['scatter'].return_value = None
        result = self.ft._receive_ft_actions()
        self.dist_mocks['scatter'].assert_called_once_with(
            ANY,
            scatter_list=None,
            src=0,
            group=FaultTolerance._recovery_group
        )
        self.assertTrue(torch.equal(result, torch.tensor([0])))

    def test_generate_scheduler_output_key_normal(self):
        """正常生成 key。"""
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
        """处理空列表的情况。"""
        mock_out = MagicMock()
        mock_out.scheduled_new_reqs = []
        mock_cached = MagicMock()
        mock_cached.req_ids = []
        mock_cached.num_computed_tokens = []
        mock_out.scheduled_cached_reqs = mock_cached

        expected_data = {
            'new_req_ids': [],
            'cached_req_ids': [],
            'cached_num_tokens': []
        }
        expected_json = json.dumps(expected_data, sort_keys=True, ensure_ascii=False)
        expected_hash = hashlib.sha256(expected_json.encode('utf-8')).hexdigest()

        result = self.ft._generate_scheduler_output_key(mock_out)
        self.assertEqual(result, expected_hash)

    def test_create_essential_state_backup_full(self):
        """完整备份验证。"""
        # 构造 model_runner 模拟对象
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
        ib.block_table.block_tables = [MagicMock(num_blocks_per_row=[4,5])]
        ib.batch_update_builder._removed = [6]
        ib.batch_update_builder.added = [7]
        ib.token_ids_cpu = [8]
        ib.num_tokens = 9
        ib.num_tokens_no_spec = 10
        ib.num_computed_tokens_cpu = [11]
        ib.num_accepted_tokens_cpu = [12]
        ib.prev_sampled_token_ids = torch.tensor([13])

        self.model_runner.eplb_updator = MagicMock(
            update_info_all={'info': 'test'},
            reqs=['r1'],
            cur_iterations=2
        )

        backup = self.ft._create_essential_state_backup('arg', kw='val')

        # 验证 args/kwargs
        self.assertEqual(backup['args'], ('arg',))
        self.assertEqual(backup['kwargs'], {'kw': 'val'})

        # 验证 generator_state
        self.assertTrue(torch.equal(backup['generator_state'], torch.tensor([1,2,3])))

        # 验证 requests
        req_backup = backup['requests_essential']['req1']
        self.assertEqual(req_backup['output_token_ids'], [1,2,3])
        self.assertEqual(req_backup['num_computed_tokens'], 5)
        self.assertEqual(len(req_backup['block_ids']), 2)

        # 验证 input_batch
        self.assertEqual(backup['_req_ids'], ['r1'])
        self.assertEqual(backup['req_output_token_ids'], [[1], [2]])
        self.assertEqual(backup['req_id_to_index'], {'r1': 0})
        self.assertEqual(backup['spec_token_ids'], [[3]])
        self.assertEqual(backup['num_blocks_per_row'], [[4,5]])
        self.assertEqual(backup['_removed'], [6])
        self.assertEqual(backup['added'], [7])
        self.assertEqual(backup['token_ids_cpu'], [8])
        self.assertEqual(backup['num_tokens'], 9)
        self.assertEqual(backup['num_tokens_no_spec'], 10)
        self.assertEqual(backup['num_computed_tokens_cpu'], [11])
        self.assertEqual(backup['num_accepted_tokens_cpu'], [12])
        self.assertTrue(torch.equal(backup['prev_sampled_token_ids'], torch.tensor([13])))

        # 验证 eplb
        self.assertEqual(backup['update_info_all'], {'info': 'test'})
        self.assertEqual(backup['reqs'], ['r1'])
        self.assertEqual(backup['cur_iterations'], 2)

    def test_restore_essential_state_full(self):
        """完整恢复验证。"""
        # 构造备份数据
        backup = {
            'args': ('arg',),
            'kwargs': {'kw': 'val'},
            'generator_state': torch.tensor([99]),
            'requests_essential': {
                'req1': {
                    'output_token_ids': [4,5,6],
                    'num_computed_tokens': 7,
                    'block_ids': (MagicMock(), MagicMock())
                }
            },
            '_req_ids': ['r2'],
            'req_output_token_ids': [[8]],
            'req_id_to_index': {'r2': 1},
            'spec_token_ids': [[9]],
            'num_blocks_per_row': [[10,11]],
            '_removed': [12],
            'added': [13],
            'token_ids_cpu': [14],
            'num_tokens': 15,
            'num_tokens_no_spec': 16,
            'num_computed_tokens_cpu': [17],
            'num_accepted_tokens_cpu': [18],
            'prev_sampled_token_ids': torch.tensor([19]),
            'update_info_all': {'info2': 'test2'},
            'reqs': ['r2'],
            'cur_iterations': 3
        }

        # 构造 model_runner 模拟对象
        self.model_runner.requests = {
            'req1': MagicMock()
        }
        self.model_runner.input_batch = MagicMock()
        ib = self.model_runner.input_batch
        ib.block_table.block_tables = [MagicMock(num_blocks_per_row=[0,0])]
        ib.batch_update_builder = MagicMock()
        ib.batch_update_builder._removed = []
        ib.batch_update_builder.added = []
        ib.token_ids_cpu = [0]
        ib.num_tokens = 0
        ib.num_tokens_no_spec = 0
        ib.num_computed_tokens_cpu = [0]
        ib.num_accepted_tokens_cpu = [0]
        ib.prev_sampled_token_ids = torch.tensor([0])

        self.model_runner.eplb_updator = MagicMock()

        # 执行恢复
        self.ft._restore_essential_state(backup)

        # 验证 RNG
        self.npu_mocks['set_rng_state'].assert_called_once_with(torch.tensor([99]))

        # 验证 requests
        req_state = self.model_runner.requests['req1']
        self.assertEqual(req_state.output_token_ids, [4,5,6])
        self.assertEqual(req_state.num_computed_tokens, 7)
        self.assertEqual(req_state.block_ids, backup['requests_essential']['req1']['block_ids'])

        # 验证 input_batch 属性赋值
        self.assertEqual(ib._req_ids, ['r2'])
        self.assertEqual(ib.req_output_token_ids, [[8]])
        self.assertEqual(ib.req_id_to_index, {'r2': 1})
        self.assertEqual(ib.spec_token_ids, [[9]])
        self.assertEqual(ib.block_table.block_tables[0].num_blocks_per_row, [10,11])
        self.assertEqual(ib.batch_update_builder._removed, [12])
        self.assertEqual(ib.batch_update_builder.added, [13])
        self.assertEqual(ib.token_ids_cpu, [14])
        self.assertEqual(ib.num_tokens, 15)
        self.assertEqual(ib.num_tokens_no_spec, 16)
        self.assertEqual(ib.num_computed_tokens_cpu, [17])
        self.assertEqual(ib.num_accepted_tokens_cpu, [18])
        self.assertTrue(torch.equal(ib.prev_sampled_token_ids, torch.tensor([19])))

        # 验证 eplb
        eplb = self.model_runner.eplb_updator
        self.assertEqual(eplb.update_info_all, {'info2': 'test2'})
        self.assertEqual(eplb.reqs, ['r2'])
        self.assertEqual(eplb.cur_iterations, 3)

    def test_restore_essential_state_empty_backup(self):
        """备份为空时不应做任何事。"""
        self.ft._restore_essential_state({})
        self.npu_mocks['set_rng_state'].assert_not_called()

    def test_single_node_decision(self):
        """_single_node_decision 逻辑。"""
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