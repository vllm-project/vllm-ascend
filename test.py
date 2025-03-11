import os
import torch
import torch_npu


device_id = 0

def _device_id_to_physical_device_id(device_id: int) -> int:
    if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
        if device_ids == [""]:
            raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES is set to empty"
                               "string, which means Ascend NPU support is"
                               "disabled.")
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id
    

physical_device_id = _device_id_to_physical_device_id(device_id)
print("physical_device_id: " + str(physical_device_id))

# return torch.npu.get_device_name(physical_device_id)
torch.npu.get_device_name(device_id)

for k, v in os.environ.items():
    if k == "ASCEND_RT_VISIBLE_DEVICES":
        print(k)
        print(v)
