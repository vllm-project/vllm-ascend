from vllm_ascend import ops

FUSION_OP_REGISTERED = False

try:
    ops.register_dummy_fusion_op()
    FUSION_OP_REGISTERED = True
except Exceptioe:
    print("Register dummy fusion op failed!")