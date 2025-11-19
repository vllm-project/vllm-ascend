from queue import Queue

from common import FaultToleranceLevel
class RecoveryContext:
    def __init__(self,model,exception : 'Exception',memory_block_info :'MemoryBlockInfo',rank: int,level:FaultToleranceLevel,
                 model_or_path:'str',fault_queue:'Queue'):
        self.model = model
        self.exception = exception
        self.memory_block_info = memory_block_info
        self.rank = rank
        self.level = level
        self.model_or_path = model_or_path
        self.fault_queue = fault_queue

