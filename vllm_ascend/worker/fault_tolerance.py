import queue
from memory_block_info import MemoryBlockInfo
from vllm_ascend.worker.recovery_context import RecoveryContext
from vllm_ascend.worker.recovery_strategy import FaultStatus

class FaultToleranceLevel(Enum):
    """容错级别配置"""
    OFF = 0      # 关闭容错
    BASIC = 1    # 基础容错（KV Cache UCE不恢复）
    FULL = 2     # 完整容错（KV Cache实时备份恢复）

class FaultTolerance:
    """容错管理主类"""

    def __init__(self, level: FaultToleranceLevel = FaultToleranceLevel.OFF):
        self.level = level
        self.fault_queue = queue.Queue()

        # 初始化内存信息
        self.memory_info = MemoryBlockInfo()

        # 构建责任链
        self.chain = ForceStopHandler()
        self.chain.set_next(UCEHandler(self.memory_info, self.uce_classifier))

        self.current_batch_context = None

        logger.info(f"FaultTolerance initialized with level: {level.name}")

    def fault_tolerance_decorator(self, func: Callable) -> Callable:
        """容错装饰器"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Level 0: 直接执行，不进行容错
            if self.level == FaultToleranceLevel.OFF:
                return func(*args, **kwargs)

            while True:  # 无限循环，直到成功或决定退出
                try:
                    output = func(*args,**kwargs)
                    # 执行原始函数
                    return output
                except Exception as e:
                    logger.error(f"Exception caught in decorated function: {e}")
                    recovery_context = RecoveryContext(e,memory_block_info=self.memory_info)
                    should_continue = self._handle_exception(recovery_context)

                    # 根据结果决定是否继续循环
                    if not should_continue:
                        return None

        return wrapper

    def _handle_exception(self, recovery_context: RecoveryContext) -> bool:
        """统一异常处理逻辑，返回是否继续循环"""
        rank = dist.get_rank() if dist.is_initialized() else 0

        try:
            # 1. 责任链处理异常（入队逻辑在handler内部完成）
            local_recovery_status = self.chain.handle(
                recovery_context
            )
            logger.info(f"Rank {rank} local recovery status: {local_recovery_status.name}")

            # 3. 收集所有rank恢复状态
            global_decision = self._coordinate_recovery(local_recovery_status)
            logger.info(f"Rank {rank} global decision: {global_decision.name}")

            # 4. 根据全局决策执行动作
            return self._execute_global_decision(global_decision)

        except Exception as inner_e:
            logger.error(f"Error during exception handling: {inner_e}")
            # 严重错误，退出循环
            return False
        finally:
            self.current_batch_context = None

    def _coordinate_recovery(self, local_status: RecoveryStatus) -> GlobalDecision:
        """分布式协调恢复状态，使用scatter操作分发不同指令"""
        if not dist.is_initialized():
            # 单机模式，直接根据本地状态决策
            if local_status == RecoveryStatus.SUCCESS:
                return GlobalDecision.RETRY_TOKEN
            else:
                return GlobalDecision.THROW_EXCEPTION

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 1. 将本地状态转换为tensor
        status_tensor = torch.tensor([local_status.value], dtype=torch.int32)

        # 2. 收集所有rank的状态
        all_statuses = torch.zeros(world_size, dtype=torch.int32)
        dist.all_gather_into_tensor(all_statuses, status_tensor)

        # 3. Rank0分析全局状态
        if rank == 0:
            # 转换回RecoveryStatus
            all_recovery_statuses = [
                RecoveryStatus(int(val)) for val in all_statuses.tolist()
            ]

            # 分析全局状态
            success_ranks = []
            failure_ranks = []

            for i, status in enumerate(all_recovery_statuses):
                if status == RecoveryStatus.SUCCESS:
                    success_ranks.append(i)
                else:
                    failure_ranks.append(i)

            # 决策逻辑
            decisions = [GlobalDecision.RETURN] * world_size  # 默认RETURN

            if len(failure_ranks) == 0:  # 全部成功
                # 所有rank都重推
                for i in range(world_size):
                    decisions[i] = GlobalDecision.RETRY_TOKEN
            elif len(success_ranks) == 0:  # 全部失败
                # 所有rank都抛出异常
                for i in range(world_size):
                    decisions[i] = GlobalDecision.THROW_EXCEPTION
            else:  # 部分失败
                # 成功的rank RETURN，失败的rank THROW_EXCEPTION
                for i in success_ranks:
                    decisions[i] = GlobalDecision.RETURN
                for i in failure_ranks:
                    decisions[i] = GlobalDecision.THROW_EXCEPTION

            # 创建scatter buffer
            decision_tensors = [torch.tensor([decisions[i].value], dtype=torch.int32) for i in range(world_size)]
            # Rank0发送自己的决策
            recv_tensor = torch.tensor([decisions[rank].value], dtype=torch.int32)

            # Scatter决策
            dist.scatter(recv_tensor, scatter_list=decision_tensors if rank == 0 else None, src=0)
            return GlobalDecision(recv_tensor.item())
        else:
            # 非Rank0接收scatter
            recv_tensor = torch.tensor([0], dtype=torch.int32)
            dist.scatter(recv_tensor, scatter_list=None, src=0)
            return GlobalDecision(recv_tensor.item())

    def _execute_global_decision(self, decision: GlobalDecision) -> bool:
        """根据全局决策执行相应动作，返回是否继续循环"""
        if decision == GlobalDecision.RETRY_TOKEN:
            logger.info("Global decision: RETRY_TOKEN - will retry current batch in loop")
            # 继续循环，重新执行func(*args, **kwargs)
            return True
        elif decision == GlobalDecision.THROW_EXCEPTION:
            logger.warning("Global decision: THROW_EXCEPTION - re-raising exception")
            if self.current_batch_context and 'exception' in self.current_batch_context:
                raise self.current_batch_context['exception']
            else:
                raise RuntimeError("Recovery failed, aborting request")
        elif decision == GlobalDecision.RETURN:
            logger.info("Global decision: RETURN - terminating inference loop")
            # 退出循环
            return False
        else:
            logger.error(f"Unknown global decision: {decision}")
            # 未知决策，退出循环
            return False