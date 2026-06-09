// S3FP16 variant of the Qwen3.6 hybrid mega kernel: gate_up cube emits fp16 (per-channel
// w13 scale folded into FIXPIPE), gu_ws is half, S3 drops the per-channel TCOLEXPANDMUL.
#define I_DIM_OVERRIDE 128
#include "mega_moe_w4a4_pto-isa.cpp"
