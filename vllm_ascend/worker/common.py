import torch
class FaultStatus():
    """
    异常状态，由fault_tolerance入队fault_queue，并由fault_aware取出
    """
    ACTIVE = torch.tensor([0])
    UCE_ERR = torch.tensor([1])
    FORCE_STOP = torch.tensor([2])
    NETWORK_ERR = torch.tensor([3])

class FaultCommand():
    """
    故障同步指令，fault_aware中rank0发出
    """
    INIT_CMD = torch.tensor([0])
    SILENCE_CMD = torch.tensor([1])
    STOP_DEVICE_CMD = torch.tensor([2])

class UCEType():
    """
    HBM UCE的具体异常类型
    """
    WEIGHTS_UCE = "WEIGHTS_UCE"
    KVCACHE_UCE = "KVCACHE_UCE"
    ACTIVATION_UCE = "ACTIVATION_UCE"
    UNKNOWN_UCE = "UNKNOWN_UCE"

class RecoveryStatus():
    """
    故障恢复的情况，由recover返回
    """
    SUCCESS_RECOMPUTE = torch.tensor([0])
    FAILED_ABORT = torch.tensor([1])

class FaultAction():
    """
    请求处理的手段，重推、终止(abort)或抛出异常给上层
    """
    RAISE_EXCEPTION = torch.tensor([0])
    RETURN = torch.tensor([1])
    RECOMPUTE = torch.tensor([2])