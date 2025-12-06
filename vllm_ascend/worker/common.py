import torch
from enum import Enum

class FaultToleranceLevel(Enum):
    """
    Fault tolerance level
    level 0: disable fault tolerance
    level 1: enable base fault tolerance for weight UCE/Activation UCE/Network Error
    level 2: enable all fault tolerance for weight UCE/Activation UCE/KVCache UCE/Network Error
    """
    OFF = 0
    BASIC = 1
    FULL = 2

class FaultStatus(Enum):
    """
    Fault status which fault_tolerance put into fault_queue
    """
    ACTIVE = torch.tensor([0])
    UCE_ERR = torch.tensor([1])
    FORCE_STOP = torch.tensor([2])
    NETWORK_ERR = torch.tensor([3])

class FaultCommand:
    """
    Fault command which rank 0 broadcast in fault_aware
    """
    INIT_CMD = torch.tensor([0])
    SILENCE_CMD = torch.tensor([1])
    STOP_DEVICE_CMD = torch.tensor([2])

class UCEType(Enum):
    """
    Specific uce type for HBM UCE
    """
    WEIGHTS_UCE = "WEIGHTS_UCE"
    KVCACHE_UCE = "KVCACHE_UCE"
    ACTIVATION_UCE = "ACTIVATION_UCE"
    UNKNOWN_UCE = "UNKNOWN_UCE"

class RecoveryStatus:
    SUCCESS = torch.tensor([0])
    FAILED = torch.tensor([1])

class FaultAction:
    RAISE_EXCEPTION = torch.tensor([0])
    RETURN = torch.tensor([1])
    RECOMPUTE = torch.tensor([2])