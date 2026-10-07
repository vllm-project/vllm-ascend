import os

import torch
from vllm.platforms import current_platform


def main() -> None:
    user_device_id = 0
    torch.npu.set_device(user_device_id)
    physical_device_id = current_platform.visible_device_id_to_physical_device_id(user_device_id)

    print(f"ASCEND_RT_VISIBLE_DEVICES={os.getenv('ASCEND_RT_VISIBLE_DEVICES')}")
    print(f"user device ID: {user_device_id}")
    print(f"host physical device ID: {physical_device_id}")


if __name__ == "__main__":
    main()
