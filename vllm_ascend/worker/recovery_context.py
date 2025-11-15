class RecoveryContext:
    def __init__(self,exeception : 'Exception',memory_block_info :'MemoryBlockInfo',rank: int):
        self.exeception = exeception
        self.memory_block_info = memory_block_info
        self.rank = rank

