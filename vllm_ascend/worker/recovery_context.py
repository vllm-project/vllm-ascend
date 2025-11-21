from queue import Queue
from vllm_ascend.worker.common import FaultToleranceLevel
from vllm_ascend.worker.memory_block_info import MemoryBlockInfo
class RecoveryContext:
    def __init__(self,model,level:FaultToleranceLevel,exception : 'Exception',rank: int,model_or_path:'str',
                 memory_block_info :'MemoryBlockInfo',fault_queue:'Queue'):
        self.model = model
        self.level = level
        self.exception = exception
        self.rank = rank
        self.model_or_path = model_or_path
        self.memory_block_info = memory_block_info
        self.fault_queue = fault_queue

