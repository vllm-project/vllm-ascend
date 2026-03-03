import queue
import threading
import time
from datetime import timedelta

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.token_reinference.common import FaultCommand, FaultStatus


class FaultAware:
    _fault_aware_group = None

    def __init__(
        self,
        rank: int,
        world_size: int,
        fault_queue: queue.Queue,
        interval_s: int = 1,
        aware_event: threading.Event | None = None,
        stop_event: threading.Event | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.npu_id = torch.npu.current_device()
        self.fault_queue = fault_queue
        self.interval_s = interval_s

        self._fault_aware_thread: threading.Thread | None = None
        self.aware_event = aware_event
        self.stop_event = stop_event

    def init_fault_aware_group(self):
        """
        Initialize the Torch process group for fault aware.
        Rank 0 is the coordinator rank,
        the other ranks are the normal rank,which is used for sending status to rank 0.

        Rank 0 will collect the status from all the other ranks and broadcast stop_device
        command to all the other ranks through `_fault_aware_group`
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError("Default torch process group must be initialized")

        if not torch.distributed.is_gloo_available():
            raise RuntimeError("Gloo backend must be available")

        logger.info("init fault aware process group: rank=%s,world_size=%s,backend=gloo", self.rank, self.world_size)
        try:
            FaultAware._fault_aware_group = torch.distributed.new_group(
                ranks=None, timeout=timedelta(minutes=5), backend="gloo"
            )
            logger.info("Rank %s successfully initialized fault aware process group", self.rank)
        except Exception as e:
            logger.error("Rank %s failed to initialize fault aware group:%s", self.rank, e)
            raise e

    def start(self):
        """Start the fault aware"""
        if self._fault_aware_thread is not None and self._fault_aware_thread.is_alive():
            logger.warning("Fault aware thread is already running")
            return
        self.init_fault_aware_group()
        logger.info("Rank %s starting fault aware thread", self.rank)
        try:
            self._fault_aware_thread = threading.Thread(
                target=self._handler_loop,
                name=f"FaultAware-Rank{self.rank}",
                daemon=True,
            )
            assert self._fault_aware_thread is not None
            self._fault_aware_thread.start()
            logger.info("Rank %s successfully started fault aware thread", self.rank)
        except Exception as e:
            logger.error("Rank %s failed to start fault aware thread:%s", self.rank, e)
            raise e

    def _handler_loop(self):
        current_status = FaultStatus.ACTIVE.value
        status_list = [torch.zeros_like(current_status) for _ in range(self.world_size)] if self.rank == 0 else None
        while True:
            try:
                current_status = self._update_status_from_queue(current_status)
                self._gather_statuses(current_status, status_list)
                fault_cmd = self._determine_fault_command(status_list)
                self.broadcast_command(fault_cmd)
                current_status = self._execute_command(fault_cmd, current_status)
            except Exception as e:
                logger.error("Exception in fault aware handler:%s", e)
                if not threading.main_thread().is_alive():
                    break
                raise e
        logger.info("Fault aware handler exiting")

    def _update_status_from_queue(self, current_status):
        try:
            msg = self.fault_queue.get_nowait()
            if msg:
                logger.info("Received new status: %s,updating status", msg.name)
                current_status = msg.value
        except queue.Empty:
            if not threading.main_thread().is_alive():
                raise RuntimeError("Main thread is not alive")
        except Exception as e:
            logger.error("Error reading from fault queue:%s", e)
            raise e

        return current_status

    def _gather_statuses(self, current_status, status_list):
        """Gather statuses from all ranks to rank 0"""
        try:
            torch.distributed.gather(
                tensor=current_status,
                gather_list=status_list,
                dst=0,
                group=FaultAware._fault_aware_group,
            )
        except Exception as e:
            logger.error("Rank %s failed to gather status:%s", self.rank, e)
            raise e

    def _determine_fault_command(self, status_list):
        """Determine the command to run"""
        fault_cmd = FaultCommand.INIT_CMD
        if self.rank == 0:
            if all(torch.equal(t, FaultStatus.ACTIVE.value) for t in status_list):
                fault_cmd = FaultCommand.SILENCE_CMD
            else:
                fault_cmd = FaultCommand.STOP_DEVICE_CMD
        return fault_cmd

    def broadcast_command(self, fault_cmd):
        """BroadCast the fault command to all ranks"""
        try:
            torch.distributed.broadcast(
                tensor=fault_cmd,
                src=0,
                group=FaultAware._fault_aware_group,
            )
        except Exception as e:
            logger.error("Rank %s failed to broadcast command:", self.rank, e)
            raise e

    def _execute_command(self, fault_cmd, current_status):
        """Execute the fault command"""
        if torch.equal(fault_cmd, FaultCommand.SILENCE_CMD):
            time.sleep(self.interval_s)
        elif torch.equal(fault_cmd, FaultCommand.STOP_DEVICE_CMD):
            logger.info("Error detected in cluster,executing stop_device on NPU %s", self.npu_id)
            self._stop_device()
            current_status = FaultStatus.ACTIVE.value
        else:
            logger.error("Unknown fault command received:%s", fault_cmd)

        return current_status

    def _stop_device(self):
        try:
            torch_npu.npu.stop_device(self.npu_id)
            if self.stop_event:
                logger.info("NPU %s execute stop device", self.npu_id)
                self.stop_event.set()

            if self.aware_event:
                logger.info("Waiting for recovery event")
                self.aware_event.wait()
                self.aware_event.clear()
                logger.info("Recovery event received,resuming operation")
        except Exception as e:
            logger.error("Error during stop_device or recovery:%s", e)
            raise e
