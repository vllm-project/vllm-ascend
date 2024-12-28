from vllm import PlatformRegistry


def register():
    # Register the ascend platform
    PlatformRegistry.register_platform(
        "npu", "vllm_ascend_plugin.platform.NPUPlatform")
    # Set the current platform to the ascend platform
    PlatformRegistry.set_current_platform("ascend")
