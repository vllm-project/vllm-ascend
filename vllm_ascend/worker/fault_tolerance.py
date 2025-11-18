import queue
from memory_block_info import MemoryBlockInfo
from vllm_ascend.worker.common import FaultAction
from vllm_ascend.worker.recovery_chain import UCEHandler
from vllm_ascend.worker.recovery_context import RecoveryContext
from vllm_ascend.worker.recovery_strategy import FaultStatus

class FaultToleranceLevel(Enum):
    """
    Fault tolerance level
    level 0: disable fault tolerance
    level 1: enable base fault tolerance for weight UCE/Activation UCE/Network Error
    level 2: enable all fault tolerance for weight UCE/Activation UCE/KVCache UCE/Network Error
    """
    OFF = 0      # 关闭容错
    BASIC = 1    # 基础容错（KV Cache UCE不恢复）
    FULL = 2     # 完整容错（KV Cache实时备份恢复）

class FaultTolerance:
    _recovery_group = None
    def __init__(self, level: FaultToleranceLevel = FaultToleranceLevel.OFF):
        self.level = level
        self.fault_queue = queue.Queue()
        self.memory_info = None
        self.recovery_chain = self._build_recovery_chain()

        # 分布式属性初始化
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self._init_recovery_group()
    #TODO: 不同并行策略下，通信组的初始化方式待调整（所需参数待确认）
    def _init_recovery_group(self):
        """初始化恢复专用进程组（与FaultAware类似）"""
        if not dist.is_initialized() or self.world_size == 1:
            return

        logger.info(
            f"Initializing recovery process group: "
            f"rank={self.rank}, world_size={self.world_size}, backend=gloo"
        )

        # 创建恢复专用进程组
        FaultTolerance._recovery_group = dist.new_group(
            ranks=None,  # 包含所有rank
            timeout=timedelta(minutes=5),
            backend="gloo",  # 使用gloo后端
        )

        logger.info("Recovery process group initialized successfully")

    def _build_recovery_chain(self) -> RecoveryHandler:
        """initialize recovery chain"""
        force_stop_handler = ForceStopHandler()
        network_handler = NetworkHandler()
        uce_handler = UCEHandler()

        force_stop_handler.set_next(network_handler).set_next(uce_handler)

        return force_stop_handler

    def fault_tolerance_decorator(self, func: Callable) -> Callable:
        """fault tolerance decorator"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # disable fault tolerance
            if self.level == FaultToleranceLevel.OFF:
                return func(*args, **kwargs)
            # enable fault tolerance
            while True:
                try:
                    output = func(*args, **kwargs)
                    return output
                except Exception as e:
                    recovery_context = RecoveryContext(
                        e,
                        memory_block_info=self.memory_info,
                        rank=self.rank,
                        level=self.level,
                        fault_queue=self.fault_queue
                    )
                    ft_action = self._handle_exception(recovery_context)
                    if torch.equal(ft_action,FaultAction.RECOMPUTE):
                        continue
                    elif torch.equal(ft_action,FaultAction.RAISE_EXCEPTION):
                        raise e
                    elif torch.equal(ft_action,FaultAction.FAILED_ABORT):
                        return None
                    else:
                        raise e

        return wrapper

    def _handle_exception(self, ctx: RecoveryContext) -> torch.Tensor:
        try:
            # 1. 责任链处理异常,并返回故障恢复状态
            local_recovery_status = self.recovery_chain.handle(ctx)  # 返回Tensor

            # 2. 故障恢复状态上报，请求决策获取
            ft_action = self._coordinate_recovery(local_recovery_status)

            # 3. 返回请求处理操作
            return ft_action

        except Exception as inner_e:
            logger.error(f"Error in exception handling: {inner_e}")
            return False

    def _coordinate_recovery(self, local_status: torch.Tensor) -> torch.Tensor:
        """
        Rank 0 Gather Recovery Status and decide global fault action
        """
        if not dist.is_initialized() or self.world_size == 1:
            return self._single_node_decision(local_status)

        # 确保Tensor在NPU上
        local_tensor = local_status.npu() if local_status.device.type != 'npu' else local_status

        # 收集所有rank的状态
        all_status_tensors = self._gather_recovery_statuses(local_tensor)

        if self.rank == 0:
            decisions = self._analyze_global_status(all_status_tensors)
            return self._scatter_decisions(decisions)
        else:
            return self._receive_decision()

    def _single_node_decision(self, local_status: torch.Tensor) -> torch.Tensor:
        """单机决策"""
        if torch.equal(local_status, RecoveryStatus.SUCCESS_RECOMPUTE):
            return FaultAction.RECOMPUTE
        else:
            return FaultAction.RAISE_EXCEPTION

    def _gather_recovery_statuses(self, local_tensor: torch.Tensor) -> List[torch.Tensor]:
        """使用gather收集恢复状态"""
        if self.rank == 0:
            # Rank 0准备接收缓冲区
            gather_list = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
            dist.gather(
                local_tensor,
                gather_list=gather_list,
                dst=0,
                group=FaultTolerance._recovery_group
            )
            return gather_list
        else:
            # 其他rank只发送，不接收
            dist.gather(local_tensor, gather_list=None, dst=0)
            return []  # 非rank0返回空列表

    def _analyze_global_status(self, all_status_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Analyze global status and generate decisions
        """
        success_ranks = []
        failure_ranks = []

        for rank, status_tensor in enumerate(all_status_tensors):
            if torch.equal(status_tensor, RecoveryStatus.SUCCESS_RECOMPUTE):
                success_ranks.append(rank)
            elif torch.equal(status_tensor, RecoveryStatus.FAILED_ABORT):
                failure_ranks.append(rank)
            else:
                logger.warning(f"Unknown status tensor from rank {rank}: {status_tensor}")
                failure_ranks.append(rank)

        logger.info(f"Global recovery: {len(success_ranks)} success, {len(failure_ranks)} failure")

        decisions = []
        if not failure_ranks:  # 全部恢复成功，执行重推
            logger.info("All ranks recovered, issuing RECOMPUTE to all")
            decisions = [FaultAction.RECOMPUTE] * self.world_size
        elif not success_ranks:  # 全部恢复失败，均抛出异常
            logger.warning("All ranks failed, issuing RAISE_EXCEPTION to all")
            decisions = [FaultAction.RAISE_EXCEPTION] * self.world_size
        else:  # 一部分恢复成功，一部分恢复失败
            logger.warning(f"Partial recovery - success ranks: {success_ranks}")
            for rank in range(self.world_size):
                if rank in success_ranks:
                    decisions.append(FaultAction.RETURN)
                else:
                    decisions.append(FaultAction.RAISE_EXCEPTION)

        return decisions

    def _scatter_decisions(self, decisions: List[torch.Tensor]) -> torch.Tensor:
        """分发决策"""
        # 确保所有决策Tensor在NPU上
        decisions_tensors = [decision.npu() for decision in decisions]

        # Rank 0分发决策
        recv_tensor = torch.tensor([0], dtype=torch.int32, device='npu')
        dist.scatter(
            recv_tensor,
            scatter_list=decisions_tensors,
            src=0,
            group=FaultTolerance._recovery_group
        )
        return recv_tensor

    def _receive_decision(self) -> torch.Tensor:
        """非Rank 0接收决策"""
        recv_tensor = torch.tensor([0], dtype=torch.int32, device='npu')
        dist.scatter(
            recv_tensor,
            scatter_list=None,
            src=0,
            group=FaultTolerance._recovery_group
        )
        return recv_tensor

    def _execute_global_decision(self, decision: torch.Tensor, ctx: RecoveryContext) -> bool:
        """根据全局决策执行相应动作，返回是否继续循环"""
        if torch.equal(decision, FaultAction.RECOMPUTE):
            logger.info("Retrying current batch with RECOMPUTE")
            return True
        elif torch.equal(decision, FaultAction.RAISE_EXCEPTION):
            logger.warning("Raising exception due to recovery failure")
            if ctx is not None and ctx.exception is not None:
                raise ctx.exception
            else:
                raise RuntimeError("No exception in RecoveryContext")
        elif torch.equal(decision, FaultAction.RETURN):
            logger.info("Terminating inference loop with RETURN")
            return False
        else:
            logger.error(f"Unknown decision: {decision}")
            return False

    def destroy_recovery_group(self):
        """销毁恢复进程组（与FaultAware对称）"""
        if FaultTolerance._recovery_group is None:
            return

        logger.info("Destroying recovery process group")
        try:
            dist.destroy_process_group(FaultTolerance._recovery_group)
            FaultTolerance._recovery_group = None
            logger.info("Successfully destroyed recovery process group")
        except Exception as e:
            logger.error(f"Failed to destroy recovery process group: {e}")

    def _get_current_weight_memory_info(self):
        memory_block = {}
        state_dict = self.model_runner.model.state_dict()
        for name,param in state_dict.items():
            start_address = param.data_ptr()
            size_bytes = param.numel() * param.element_size()
            end_address = start_address + max(0,size_bytes-1)
            memory_block[name] = {
                'name':name,
                'start_address':start_address,
                'end_address':end_address,
            }
        return memory_block