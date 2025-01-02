def register():
    """Register the NPU platform."""
    # register custom ops
    return "vllm_ascend_plugin.platform.NPUPlatform"
