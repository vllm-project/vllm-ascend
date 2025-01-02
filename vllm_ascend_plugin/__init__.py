def register():
    from vllm_ascend_plugin import ops
    """Register the NPU platform."""
    # register custom ops
    return "vllm_ascend_plugin.platform.NPUPlatform"
