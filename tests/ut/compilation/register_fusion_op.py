from vllm_ascend import ops
FUSION_OP_REGISTERED = False

try:
    ops.register_dummy_fusion_op()
    FUSION_OP_REGISTERED = True
except Exception as e:
    print("Regist dummy fusion op failed!")