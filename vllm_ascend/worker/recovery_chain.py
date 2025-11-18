from common import FaultStatus,UCEType
from vllm_ascend.worker.recovery_context import RecoveryContext

all_err = [
    "UCE ERROR",
    "HBM MULTI BIT ECC ERROR",
    "FORCE STOP",
    "SUSPECT MEM ERROR",
    "HCCS LINK ERROR",
    "HCCL OP RETRY FAILED",
    "SUSPECT REMOTE ERROR"
]


uce_error = ["UCE ERROR"]
force_stop_error = ["FORCE STOP"]
network_error = [
    "SUSPECT REMOTE ERROR",
    "HCCS LINK ERROR",
    "HCCL OP RETRY FAILED"
]


class RecoveryHandler(ABC):
    """责任链处理器基类"""
    def __init__(self):
        self.next_handler = None

    def set_next(self, handler: 'RecoveryHandler') -> 'RecoveryHandler':
        """Set next handler"""
        self.next_handler = handler
        return handler

    @abstractmethod
    def can_handle(self, ctx:RecoveryContext) -> bool:
        pass

    @abstractmethod
    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:
        """Specific recovery function"""
        pass

    def handle(self, ctx:RecoveryContext) -> RecoveryStatus:
        """ Entry point for RecoveryHandler """
        if self.can_handle(ctx):
            return self.recover(ctx)
        elif self.next_handler:
            return self.next_handler.handle(ctx)
        else:
            logger.warning("No handler can process the exception")
            return RecoveryStatus.FAILURE


class ForceStopHandler(RecoveryHandler):
    """处理Force Stop异常的处理器"""

    def can_handle(self, ctx:RecoveryContext) -> bool:
        """判断是否为Force Stop异常，如果是则入队"""
        error_str = str(ctx.exception).lower()
        return 'force stop' in error_str


    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:
        """处理Force Stop异常，仅返回状态"""
        return RecoveryStatus.SUCCESS_RECOMPUTE

class NetworkHandler(RecoveryHandler):

    def can_handle(self, ctx:RecoveryContext) -> bool:
        error_str = str(ctx.exception).lower()
        if 'remote' in error_str:
            ctx.fault_queue.put_nowait(FaultStatus.NETWORK_ERR)
            return true
        return false

    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:
        """恢复Network Error,无特殊操作"""
        return RecoveryStatus.SUCCESS_RECOMPUTE

class UCEHandler(RecoveryHandler):
    """统一处理UCE异常的处理器"""
    def can_handle(self, ctx:RecoveryContext) -> bool:
        """判断是否为UCE异常，如果是则入队"""
        error_str = str(ctx.exception).lower()
        if 'uce' in error_str:
            ctx.fault_queue.put_nowait(FaultStatus.UCE_ERR)
            return true
        return false

    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:
        """处理UCE异常，内部判断具体类型并执行恢复"""
        try:
            exception = ctx.exception
            if not exception:
                logger.error("Missing exception in fault context")
                return RecoveryStatus.FAILURE

            # 1. 分类UCE错误类型和地址
            error_type, address = self.uce_classifier.classify_uce_error(exception, fault_context)
            fault_context['uce_type'] = error_type.value
            fault_context['hbm_address'] = address

            logger.info(f"UCEHandler: UCE error classified as {error_type.value} at address {address}")

            # 2. 根据错误类型执行恢复
            if error_type == UCEErrorType.WEIGHT_UCE:
                return self._recover_weight_uce(fault_context, config)
            elif error_type == UCEErrorType.KV_CACHE_UCE:
                return self._recover_kv_cache_uce(fault_context, config)
            elif error_type == UCEErrorType.ACTIVATION_UCE:
                return self._recover_activation_uce(fault_context, config)
            else:
                logger.warning(f"UCEHandler: Unknown UCE type: {error_type.value}")
                return RecoveryStatus.FAILURE

        except Exception as e:
            logger.error(f"UCEHandler: UCE recovery failed: {e}")
            return RecoveryStatus.FAILURE

    def _recover_weight_uce(self, fault_context: Dict, config: Any) -> RecoveryStatus:
        """恢复权重UCE错误"""
        address = fault_context.get('hbm_address')
        if address is None:
            logger.error("UCEHandler: Missing HBM address for weight UCE recovery")
            return RecoveryStatus.FAILURE

        # 1. 通过地址映射找到层名
        layer_name = None
        for (start, end), name in self.memory_info.weight_address_map.items():
            if start <= address < end:
                layer_name = name
                break

        if not layer_name:
            logger.error(f"UCEHandler: Cannot map address {address} to layer name")
            return RecoveryStatus.FAILURE

        logger.info(f"UCEHandler: Recovering weight UCE for layer: {layer_name}")

        # 2. 增量重加载权重
        try:
            if hasattr(self.model_engine, 'reload_layer_weights'):
                success = self.model_engine.reload_layer_weights(layer_name)
                if success:
                    logger.info(f"UCEHandler: Successfully reloaded weights for layer {layer_name}")
                    return RecoveryStatus.SUCCESS
            # 尝试通用重加载
            self._generic_weight_reload(layer_name)
            return RecoveryStatus.SUCCESS
        except Exception as e:
            logger.error(f"UCEHandler: Weight reload failed: {e}")
            return RecoveryStatus.FAILURE

    def _recover_kv_cache_uce(self, fault_context: Dict, config: Any) -> RecoveryStatus:
        """恢复KV Cache UCE错误"""
        level = getattr(config, 'level', FaultToleranceLevel.OFF)

        if level == FaultToleranceLevel.BASIC:
            logger.warning("UCEHandler: KV Cache UCE in BASIC level, aborting recovery")
            return RecoveryStatus.FAILURE

        if level == FaultToleranceLevel.FULL:
            try:
                # 检查KV Cache备份是否存在
                if not hasattr(self.model_engine, 'kv_cache_backup') or self.model_engine.kv_cache_backup is None:
                    logger.error("UCEHandler: KV Cache backup not available for recovery")
                    return RecoveryStatus.FAILURE

                logger.info("UCEHandler: Recovering KV Cache from backup")
                # 从备份恢复KV Cache
                if hasattr(self.model_engine, 'restore_kv_cache_from_backup'):
                    success = self.model_engine.restore_kv_cache_from_backup()
                    if success:
                        logger.info("UCEHandler: Successfully restored KV Cache from backup")
                        return RecoveryStatus.SUCCESS
                return RecoveryStatus.FAILURE
            except Exception as e:
                logger.error(f"UCEHandler: KV Cache recovery failed: {e}")
                return RecoveryStatus.FAILURE

        logger.warning(f"UCEHandler: Unsupported fault tolerance level: {level}")
        return RecoveryStatus.FAILURE

    def _recover_activation_uce(self, fault_context: Dict, config: Any) -> RecoveryStatus:
        """恢复激活值UCE错误"""
        logger.info("UCEHandler: Activation UCE detected, no special recovery needed")
        # 激活值UCE无需特殊恢复，直接返回成功
        return RecoveryStatus.SUCCESS

    def _generic_weight_reload(self, layer_name: str):
        """通用权重重加载逻辑"""
        logger.warning(f"UCEHandler: Fallback weight reload for layer: {layer_name}")
        # 实际实现需要根据vllm模型结构
        if hasattr(self.model_engine.model, layer_name):
            layer = getattr(self.model_engine.model, layer_name)
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()