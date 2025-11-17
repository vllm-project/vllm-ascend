class MemoryBlockInfo:
    def __init__(self,model):
        self.model = model
        self.weight_blocks = {}
        self.kvcache_blocks = {}
        self.initialized = false

    def initialize(self):
        pass
    def update_weight_address(self):
        pass
    def _get_weight_memory_info(self):
        weights_blocks = {}
        state_dice = self.model_runner.model.state_dict()
        for name,param in state_dice.items():
            start_address = param.data_ptr()
            size_bytes = param.numel() * param.element_size()
            end_address = start_address + max(0,size_bytes - 1)
            weights_blocks[name] = {
                'name':name,
                'start_address':start_address,
                'end_address':end_address,
                'size_bytes':size_bytes,
                'device':param.device,
            }
        self.weight_blocks = weights_blocks
    def _load_kvcache_blocks(self):
        pass