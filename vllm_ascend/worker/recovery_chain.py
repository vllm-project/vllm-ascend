class UCEClassifier:
    """
    HBM UCE错误分类器，用于在UCEHandler中执行对应恢复手段
    """

    def __init__(self, memory_info: MemoryBlockInfo):
        self.memory_info = memory_info

    def classify_uce_error(self, exception: Exception, fault_context: Dict) -> Tuple[UCEErrorType, Optional[int]]:
        """
        分类UCE错误类型和提取地址
        返回: (错误类型, 地址)
        """
        try:
            error_str = str(exception).lower()

            # 1. 从异常信息中提取HBM地址
            address = self._extract_address_from_exception(error_str, fault_context)

            if address is None:
                logger.warning("Cannot extract HBM address from exception")
                return UCEErrorType.UNKNOWN, None

            # 2. 使用MemoryBlockInfo进行地址映射
            error_type = self.memory_info.classify_address(address)

            return error_type, address

        except Exception as e:
            logger.error(f"Error classifying UCE: {e}")
            return UCEErrorType.UNKNOWN, None

    def _extract_address_from_exception(self, error_str: str, fault_context: Dict) -> Optional[int]:
        """从异常信息中提取HBM地址"""
        # 优先从fault_context获取
        if 'hbm_address' in fault_context.get('metadata', {}):
            return int(fault_context['metadata']['hbm_address'])

        # 从异常字符串中解析地址
        import re

        # 匹配十六进制地址
        hex_match = re.search(r'0x([0-9a-fA-F]+)', error_str)
        if hex_match:
            return int(hex_match.group(0), 16)

        # 匹配十进制地址
        dec_match = re.search(r'address\s+(\d+)', error_str)
        if dec_match:
            return int(dec_match.group(1))

        return None


class RecoveryHandler:
    """责任链处理器基类"""

    def __init__(self, next_handler: Optional['RecoveryHandler'] = None):
        self.next_handler = next_handler

    def set_next(self, handler: 'RecoveryHandler') -> 'RecoveryHandler':
        """设置下一个处理器"""
        self.next_handler = handler
        return handler

    def can_handle(self, exception: Exception, fault_context: Dict, fault_queue: queue.Queue) -> bool:
        """判断是否能处理该异常，如果能处理则将状态入队"""
        raise NotImplementedError

    def recover(self, fault_context: Dict, config: Any) -> RecoveryStatus:
        """执行恢复逻辑，返回本地恢复状态"""
        raise NotImplementedError

    def handle(self, exception: Exception, fault_context: Dict, config: Any,
               fault_queue: queue.Queue) -> RecoveryStatus:
        """责任链处理入口"""
        if self.can_handle(exception, fault_context, fault_queue):
            return self.recover(fault_context, config)
        elif self.next_handler:
            return self.next_handler.handle(exception, fault_context, config, fault_queue)
        else:
            logger.warning("No handler can process the exception")
            return RecoveryStatus.FAILURE


class ForceStopHandler(RecoveryHandler):
    """处理Force Stop异常的处理器"""

    def can_handle(self, exception: Exception, fault_context: Dict, fault_queue: queue.Queue) -> bool:
        """判断是否为Force Stop异常，如果是则入队"""
        error_str = str(exception).lower()
        is_force_stop = ('force stop' in error_str or 'device stopped' in error_str or
                         'stop_device' in error_str)

        if is_force_stop:
            # 构建fault_context
            rank = dist.get_rank() if dist.is_initialized() else 0
            fault_context.update({
                'exception': exception,
                'rank': rank,
                'timestamp': time.time(),
                'fault_type': 'force_stop'
            })

            # 创建FaultStatus并入队
            fault_status = FaultStatus.from_context(fault_context)
            fault_queue.put(fault_status)
            logger.info(f"ForceStopHandler: Enqueued force stop fault status for rank {rank}")

            return True

        return False

    def recover(self, fault_context: Dict, config: Any) -> RecoveryStatus:
        """处理Force Stop异常，仅返回状态"""
        logger.info("ForceStopHandler: Detected Force Stop exception")
        # Force Stop异常不需要特殊恢复，直接返回成功状态
        return RecoveryStatus.SUCCESS


class UCEHandler(RecoveryHandler):
    """统一处理UCE异常的处理器"""

    def __init__(self, model_engine: Any, memory_info: MemoryBlockInfo, uce_classifier: UCEClassifier):
        super().__init__()
        self.memory_info = memory_info
        self.uce_classifier = uce_classifier

    def can_handle(self, exception: Exception, fault_context: Dict, fault_queue: queue.Queue) -> bool:
        """判断是否为UCE异常，如果是则入队"""
        error_str = str(exception).lower()
        is_uce = 'uce' in error_str

        if is_uce:
            # 构建fault_context
            rank = dist.get_rank() if dist.is_initialized() else 0
            fault_context.update({
                'exception': exception,
                'rank': rank,
                'timestamp': time.time(),
                'fault_type': 'uce_error'
            })

            # 创建FaultStatus并入队
            fault_status = FaultStatus.from_context(fault_context)
            fault_queue.put(fault_status)
            logger.info(f"UCEHandler: Enqueued UCE fault status for rank {rank}")

            return True

        return False

    def recover(self, fault_context: Dict, config: Any) -> RecoveryStatus:
        """处理UCE异常，内部判断具体类型并执行恢复"""
        try:
            exception = fault_context.get('exception')
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