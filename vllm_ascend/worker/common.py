import torch
class FaultStatus():
    ACTIVE = torch.tensor([0])
    UCE_ERR = torch.tensor([1])
    FORCE_STOP = torch.tensor([2])
    NETWORK_ERR = torch.tensor([3])

class FaultCommand():
    INIT_CMD = torch.tensor([0])
    SILENCE_CMD = torch.tensor([1])
    STOP_DEVICE_CMD = torch.tensor([2])

class UCEType():
    WEIGHTS_UCE = "WEIGHTS_UCE"
    KVCACHE_UCE = "KVCACHE_UCE"
    ACTIVATION_UCE = "ACTIVATION_UCE"

class RecoveryStatus():
    SUCCESS_RECOMPUTE = torch.tensor([0])
    FAILED_ABORT = torch.tensor([1])

class FaultAction():
    RAISE_EXCEPTION = torch.tensor([0])
    RETURN = torch.tensor([1])
    RECOMPUTE = torch.tensor([2])