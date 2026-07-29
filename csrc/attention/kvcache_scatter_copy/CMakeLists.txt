# The fused kernel is an arch22/arch32 implementation. A5 uses the eager
# composite fallback in dsa_ascend_ops_backend.py.
if(NOT SOC_VERSION MATCHES "ascend310p.*|ascend950")
    add_subdirectory(op_host)
endif()
