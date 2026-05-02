from enum import Enum

import torch


class FaultToleranceLevel(Enum):
    """
    Fault tolerance level
    level_0 : disable fault tolerance
    level 1 : enable base fault tolerance for L1->L2 Network Error
    """

    LEVEL_0 = 0
    LEVEL_1 = 1


class FaultStatus(Enum):
    """
    Fault status which fault_tolerance put into fault_queue
    """

    ACTIVE = torch.tensor(0)
    FORCE_STOP = torch.tensor(1)
    NETWORK_ERR = torch.tensor(2)


class FaultCommand:
    """
    Fault command which rank 0 broadcast in fault_aware
    """

    INIT_CMD = torch.tensor(0)
    SILENCE_CMD = torch.tensor(1)
    STOP_DEVICE_CMD = torch.tensor(2)


class RecoveryStatus:
    """
    Recovery status
    """

    SUCCESS = torch.tensor(0)
    FAILED = torch.tensor(1)


class FaultAction:
    RAISE_EXCEPTION = torch.tensor(0)
    RETURN = torch.tensor(1)
    RECOMPUTE = torch.tensor(2)
