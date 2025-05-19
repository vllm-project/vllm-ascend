from dataclasses import dataclass
import torch_npu

@dataclass
class NPUSocInfo:
    is_a3: bool = False

    def __post_init__(self):
        torch_npu.npu._lazy_init()
        self.soc_version = torch_npu._C._npu_get_soc_version()
        if self.soc_version in (253, 254, 255):
            self.is_a3 = True