import unittest
import queue
import threading
from unittest.mock import MagicMock, patch, call, ANY

import torch

from tests.ut.base import TestBase
from vllm_ascend.token_reinference.fault_aware import FaultAware
from vllm_ascend.token_reinference.common import FaultStatus, FaultCommand


class TestFaultAware(TestBase):
    def setUp(self):
        self.rank = 0
        self.world_size = 2
        self.fault_queue: queue.Queue = queue.Queue()
        self.aware_event = threading.Event()
        self.stop_event = threading.Event()

        patcher_current_device = patch('torch.npu.current_device', return_value=0)
        self.mock_current_device = patcher_current_device.start()
        self.addCleanup(patcher_current_device.stop)

        patcher_logger = patch('vllm_ascend.token_reinference.fault_aware.logger')
        self.mock_logger = patcher_logger.start()
        self.addCleanup(patcher_logger.stop)

        patcher_dist_init = patch('torch.distributed.is_initialized', return_value=True)
        self.mock_dist_is_initialized = patcher_dist_init.start()
        self.addCleanup(patcher_dist_init.stop)

        patcher_gloo_avail = patch('torch.distributed.is_gloo_available', return_value=True)
        self.mock_gloo_available = patcher_gloo_avail.start()
        self.addCleanup(patcher_gloo_avail.stop)

        patcher_new_group = patch('torch.distributed.new_group')
        self.mock_new_group = patcher_new_group.start()
        self.addCleanup(patcher_new_group.stop)

        patcher_gather = patch('torch.distributed.gather')
        self.mock_gather = patcher_gather.start()
        self.addCleanup(patcher_gather.stop)

        patcher_broadcast = patch('torch.distributed.broadcast')
        self.mock_broadcast = patcher_broadcast.start()
        self.addCleanup(patcher_broadcast.stop)

        patcher_stop_device = patch('torch_npu.npu.stop_device')
        self.mock_stop_device = patcher_stop_device.start()
        self.addCleanup(patcher_stop_device.stop)

        patcher_thread = patch('threading.Thread')
        self.mock_thread_class = patcher_thread.start()
        self.mock_thread_instance = MagicMock()
        self.mock_thread_class.return_value = self.mock_thread_instance
        self.addCleanup(patcher_thread.stop)

        self.fa = FaultAware(
            rank=self.rank,
            world_size=self.world_size,
            fault_queue=self.fault_queue,
            interval_s=1,
            aware_event=self.aware_event,
            stop_event=self.stop_event
        )

        FaultAware._fault_aware_group = None

    def test_init(self):
        self.assertEqual(self.fa.rank, self.rank)
        self.assertEqual(self.fa.world_size, self.world_size)
        self.assertEqual(self.fa.npu_id, 0)
        self.assertEqual(self.fa.fault_queue, self.fault_queue)
        self.assertEqual(self.fa.interval_s, 1)
        self.assertEqual(self.fa.aware_event, self.aware_event)
        self.assertEqual(self.fa.stop_event, self.stop_event)
        self.assertIsNone(self.fa._fault_aware_thread)

    def test_init_fault_aware_group_dist_not_initialized(self):
        self.mock_dist_is_initialized.return_value = False
        with self.assertRaises(RuntimeError) as ctx:
            self.fa.init_fault_aware_group()
        self.assertEqual(str(ctx.exception), "Default torch process group must be initialized")

    def test_init_fault_aware_group_gloo_not_available(self):
        self.mock_gloo_available.return_value = False
        with self.assertRaises(RuntimeError) as ctx:
            self.fa.init_fault_aware_group()
        self.assertEqual(str(ctx.exception), "Gloo backend must be available")

    def test_init_fault_aware_group_success(self):
        self.fa.init_fault_aware_group()
        self.mock_new_group.assert_called_once_with(
            ranks=None,
            timeout=ANY,
            backend='gloo'
        )
        self.assertIsNotNone(FaultAware._fault_aware_group)
        self.mock_logger.info.assert_called()

    def test_init_fault_aware_group_exception(self):
        self.mock_new_group.side_effect = Exception("new group failed")
        with self.assertRaises(Exception) as ctx:
            self.fa.init_fault_aware_group()
        self.assertEqual(str(ctx.exception), "new group failed")
        self.mock_logger.error.assert_called_once()

    def test_start_thread_already_running(self):
        self.fa._fault_aware_thread = MagicMock()
        self.fa._fault_aware_thread.is_alive.return_value = True
        self.fa.start()
        self.mock_logger.warning.assert_called_with("Fault aware thread is already running")
        self.mock_thread_class.assert_not_called()

    def test_start_success(self):
        with patch.object(self.fa, 'init_fault_aware_group') as mock_init:
            self.fa.start()
            mock_init.assert_called_once()
            self.mock_thread_class.assert_called_once_with(
                target=self.fa._handler_loop,
                name=f"FaultAware-Rank{self.rank}",
                daemon=True
            )
            self.mock_thread_instance.start.assert_called_once()

    def test_start_exception(self):
        with patch.object(self.fa, 'init_fault_aware_group') as mock_init:
            self.mock_thread_class.side_effect = Exception("thread creation failed")
            with self.assertRaises(Exception) as ctx:
                self.fa.start()
            self.assertEqual(str(ctx.exception), "thread creation failed")
            mock_init.assert_called_once()
            self.mock_logger.error.assert_called_once()

    def test_update_status_from_queue_with_message(self):
        current_status = FaultStatus.ACTIVE.value
        self.fault_queue.put(FaultStatus.NETWORK_ERR)
        new_status = self.fa._update_status_from_queue(current_status)
        self.assertEqual(new_status, FaultStatus.NETWORK_ERR.value)
        self.mock_logger.info.assert_called_once()

    def test_update_status_from_queue_empty_main_alive(self):
        current_status = FaultStatus.ACTIVE.value
        with patch('threading.main_thread') as mock_main:
            mock_main.return_value.is_alive.return_value = True
            new_status = self.fa._update_status_from_queue(current_status)
            self.assertEqual(new_status, current_status)
            self.mock_logger.info.assert_not_called()

    def test_update_status_from_queue_empty_main_dead(self):
        current_status = FaultStatus.ACTIVE.value
        with patch('threading.main_thread') as mock_main:
            mock_main.return_value.is_alive.return_value = False
            with self.assertRaises(RuntimeError) as ctx:
                self.fa._update_status_from_queue(current_status)
            self.assertEqual(str(ctx.exception), "Main thread is not alive")

    def test_update_status_from_queue_exception(self):
        current_status = FaultStatus.ACTIVE.value
        with patch.object(self.fault_queue, 'get', side_effect=Exception("queue error")):
            with self.assertRaises(Exception) as ctx:
                self.fa._update_status_from_queue(current_status)
            self.assertEqual(str(ctx.exception), "queue error")
            self.mock_logger.error.assert_called_once()

    def test_gather_statuses_normal(self):
        FaultAware._fault_aware_group = MagicMock()
        current_status = FaultStatus.ACTIVE.value
        status_list = [torch.zeros([1], dtype=torch.int64) for _ in range(self.world_size)]
        self.fa._gather_statuses(current_status, status_list)
        self.mock_gather.assert_called_once_with(
            tensor=current_status,
            gather_list=status_list,
            dst=0,
            group=FaultAware._fault_aware_group
        )

    def test_gather_statuses_exception(self):
        FaultAware._fault_aware_group = MagicMock()
        self.mock_gather.side_effect = Exception("gather failed")
        current_status = FaultStatus.ACTIVE.value
        status_list = [torch.zeros([1], dtype=torch.int64) for _ in range(self.world_size)]
        with self.assertRaises(Exception) as ctx:
            self.fa._gather_statuses(current_status, status_list)
        self.assertEqual(str(ctx.exception), "gather failed")
        self.mock_logger.error.assert_called_once()

    def test_determine_fault_command_rank0_all_active(self):
        self.fa.rank = 0
        status_list = [FaultStatus.ACTIVE.value, FaultStatus.ACTIVE.value]
        cmd = self.fa._determine_fault_command(status_list)
        self.assertTrue(torch.equal(cmd, FaultCommand.SILENCE_CMD))

    def test_determine_fault_command_rank0_not_all_active(self):
        self.fa.rank = 0
        status_list = [FaultStatus.ACTIVE.value, FaultStatus.NETWORK_ERR.value]
        cmd = self.fa._determine_fault_command(status_list)
        self.assertTrue(torch.equal(cmd, FaultCommand.STOP_DEVICE_CMD))

    def test_determine_fault_command_non_rank0(self):
        self.fa.rank = 1
        status_list = None
        cmd = self.fa._determine_fault_command(status_list)
        self.assertTrue(torch.equal(cmd, FaultCommand.INIT_CMD))

    def test_broadcast_command_normal(self):
        FaultAware._fault_aware_group = MagicMock()
        cmd = FaultCommand.STOP_DEVICE_CMD
        self.fa.broadcast_command(cmd)
        self.mock_broadcast.assert_called_once_with(
            tensor=cmd,
            src=0,
            group=FaultAware._fault_aware_group
        )

    def test_broadcast_command_exception(self):
        FaultAware._fault_aware_group = MagicMock()
        self.mock_broadcast.side_effect = Exception("broadcast failed")
        cmd = FaultCommand.STOP_DEVICE_CMD
        with self.assertRaises(Exception) as ctx:
            self.fa.broadcast_command(cmd)
        self.assertEqual(str(ctx.exception), "broadcast failed")
        self.mock_logger.error.assert_called_once()

    def test_execute_command_silence(self):
        cmd = FaultCommand.SILENCE_CMD
        current_status = FaultStatus.ACTIVE.value
        with patch('time.sleep') as mock_sleep:
            new_status = self.fa._execute_command(cmd, current_status)
            mock_sleep.assert_called_once_with(self.fa.interval_s)
        self.assertEqual(new_status, current_status)

    def test_execute_command_stop_device(self):
        cmd = FaultCommand.STOP_DEVICE_CMD
        current_status = FaultStatus.ACTIVE.value
        with patch.object(self.fa, '_stop_device') as mock_stop:
            new_status = self.fa._execute_command(cmd, current_status)
            mock_stop.assert_called_once()
        self.assertTrue(torch.equal(new_status, FaultStatus.ACTIVE.value))

    def test_execute_command_unknown(self):
        cmd = torch.tensor([999])
        current_status = FaultStatus.ACTIVE.value
        new_status = self.fa._execute_command(cmd, current_status)
        self.assertEqual(new_status, current_status)
        self.mock_logger.error.assert_called_once()

    def test_stop_device_normal(self):
        self.fa.aware_event = MagicMock()
        self.fa.stop_event = MagicMock()
        self.fa._stop_device()
        self.mock_stop_device.assert_called_once_with(self.fa.npu_id)
        self.fa.stop_event.set.assert_called_once()
        self.fa.aware_event.wait.assert_called_once()
        self.fa.aware_event.clear.assert_called_once()

    def test_stop_device_exception(self):
        self.mock_stop_device.side_effect = Exception("stop failed")
        with self.assertRaises(Exception) as ctx:
            self.fa._stop_device()
        self.assertEqual(str(ctx.exception), "stop failed")
        self.mock_logger.error.assert_called_once()

    def test_handler_loop_calls_methods_and_breaks_on_exception(self):
        with patch.object(self.fa, '_update_status_from_queue', return_value=FaultStatus.ACTIVE.value) as mock_update, \
             patch.object(self.fa, '_gather_statuses', side_effect=Exception("gather error")) as mock_gather, \
             patch.object(self.fa, '_determine_fault_command') as mock_determine, \
             patch.object(self.fa, 'broadcast_command') as mock_broadcast, \
             patch.object(self.fa, '_execute_command') as mock_execute, \
             patch('threading.main_thread') as mock_main:

            mock_main.return_value.is_alive.return_value = False

            self.fa._handler_loop()

            mock_update.assert_called_once()
            mock_gather.assert_called_once()
            mock_determine.assert_not_called()
            mock_broadcast.assert_not_called()
            mock_execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()