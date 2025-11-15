class MemoryBlockInfo:
    def __init__(self,model):
        self.model = model
        self.weight_blocks
        self.kvcache_blocks
        self.initialized

    def initialize(self):
        pass
    def update_weight_address(self):
        pass
    def _load_weights_blocks(self):
        pass
    def _load_kvcache_blocks(self):
        pass