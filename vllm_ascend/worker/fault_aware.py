import time
import threading
import torch
import queue
import torch.distributed
import torch_npu

from datetime import timedelta
from vllm.logger import logger
from vllm_ascend.worker.common import FaultStatus,FaultCommand

class FaultAware:
    _fault_aware_group = None

    def __init__(self,rank:int,world_size:int,fault_queue:queue.Queue,interval_s=1,
                 aware_event:threading.Event=None):
        self.rank = rank
        self.world_size = world_size
        self.npu_id = torch.npu.current_device()
        self.fault_queue = fault_queue
        self.interval_s = interval_s

        self._fault_aware_thread = None
        self.aware_event = aware_event
        self._stop_event = threading.Event()

    def init_fault_aware_group(self):
        """
        Initialize the Torch process group for fault aware.
        Rank 0 is the coordinator rank,
        the other ranks are the normal rank,which is used for sending status to rank 0.

        Rank 0 will collect the status from all the other ranks and broadcast stop_device
        command to all the other ranks through `_fault_aware_group`
        """
        assert(
            torch.distributed.is_initialized()
        ),"Default torch process group must be initialized"

        assert(
            torch.distributed.is_gloo_available()
        ),"Gloo process group must be available"

        rank = self.rank
        logger.info(
            f"init fault aware process group: "
            f"rank={rank},world_size={self.world_size},backend=gloo"
        )
        FaultAware._fault_aware_group = torch.distributed.new_group(
            ranks=None,
            timeout=timedelta(minutes=5),
            backend="gloo"
        )
        assert self._fault_aware_group is not None

    def start(self):
        """Start the fault aware"""
        self.init_fault_aware_group()
        logger.info("Start fault aware thread")
        try:
            self._fault_aware_thread = threading.Thread(
                target=self._handler,
                daemon=True,
            )
            self._fault_aware_thread.start()
            logger.info("Succeeded to start fault aware thread")
        except Exception as e:
            logger.error(f"Failed to start fault aware thread:{e}")

    def _handler(self):
        torch.npu.set_device(self.npu_id)
        status = FaultStatus.ACTIVE.value
        status_list = (
            [torch.zeros([1],dtype=torch.int64) for _ in range(self.world_size)]
            if self.rank == 0
            else None
        )
        while True:
            try:
                msg = self.fault_queue.get_nowait()
                if msg:
                    logger.info(f"Get abnormal status,update status {msg.name},update status")
                    status = msg.value
            except queue.Empty:
                if not threading.main_thread().is_alive():
                    return
            try:
                torch.distributed.gather(
                    tensor=status,
                    gather_list=status_list,
                    dst=0,
                    group = self._fault_aware_group,
                )
                fault_cmd = FaultCommand.INIT_CMD
                if self.rank == 0:
                    if all(torch.equal(t,FaultStatus.ACTIVE.value) for t in status_list):
                        fault_cmd = FaultCommand.SILENCE_CMD
                    else:
                        fault_cmd = FaultCommand.STOP_DEVICE_CMD

                torch.distributed.broadcast(
                    tensor=fault_cmd,
                    src=0,
                    group=self._fault_aware_group,
                )

                if torch.equal(fault_cmd,FaultCommand.SILENCE_CMD):
                    time.sleep(self.interval_s)
                elif torch.equal(fault_cmd,FaultCommand.STOP_DEVICE_CMD):
                    logger.info(f"Error in group,execute stop_device")
                    torch_npu.npu.stop_device(self.npu_id)
                    # Wait for fault_tolerance to wake me up
                    self.aware_event.wait()
                    self.aware_event.clear()
                    # Assume recover successfully
                    status = FaultStatus.ACTIVE.value
                else:
                    raise RuntimeError(f"Unknown fault command:{fault_cmd}")
            except Exception as e:
                time.sleep(self.interval_s)
                logger.error(f"Fault aware handler exception:{e}")
                if not threading.main_thread().is_alive():
                    return

    def destroy_fault_aware_group(self):
        """Destroy the Torch process group for fault aware"""
        if self._fault_aware_group is None:
            return
        logger.info("Destroy fault aware process group")
        try:
            torch.distributed.destroy_process_group(self._fault_aware_group)
            self._fault_aware_group = None
            logger.info("Succeeded to destroy fault aware process group")
        except Exception as e:
            logger.error(f"Failed to destroy fault aware process group:{e}")
