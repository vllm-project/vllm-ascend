class RecoveryContext:
    def __init__(self,exception : 'Exception',memory_block_info :'MemoryBlockInfo',rank: int,level,fault_queue:'Queue'):
        self.exception = exception
        self.memory_block_info = memory_block_info
        self.rank = rank
        self.level = level
        self.fault_queue = fault_queue

