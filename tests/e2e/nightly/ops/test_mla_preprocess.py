"""
Tests for the Mla Preprocess Operator.
Run `python tests/ops/test_mla_preprocess.py`.
"""

from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch.npu.config.allow_internal_format = True

SEED = 42

HIDDEN = 7168
MM1_OUT = 2112
Q_RMS = 1536
K_NOPE = 512
K_PE = 64
HEADS_Q = 16
Q_NOPE_DIM = 128
Q_PE_DIM = 64
Q_DIM = 192
HEAD_DIM = 64
Q_NOPE = 512

def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align

# ND -> NZ
def transdata(nd_mat: torch.Tensor, block_size: tuple = (16, 16)) -> torch.Tensor:
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, ((0, r_pad, 0, c_pad)))
    nz_mat = torch.permute(
        torch.reshape(
            nd_mat,
            (r // block_size[0], block_size[0], c // block_size[1], block_size[1]),
        ),
        [2, 0, 1, 3],
    )
    nz_mat = torch.reshape(
        nz_mat, (1, nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3])
    )
    return nz_mat.contiguous()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    return torch.cat([-second_half, first_half], dim=-1)

def apply_rope_half(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)
    return (x.float() * cos.float() + rotate_half(x.float()) * sin.float()).to(x.dtype)

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_norm = x.float() * torch.rsqrt(var + eps)
    y = x_norm * gamma.float()
    return y

def quant_per_tensor(
    x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor
) -> torch.Tensor:
    x = x / scale.float() + zp.float()
    x = x.to(torch.float16)
    x = torch.clamp(x, -128, 127)
    return torch.round(x).to(torch.int8)

def quant_per_tensor_muls(
    x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor
) -> torch.Tensor:
    x = x * scale.float() + zp.float()
    x = x.to(torch.float16)
    x = torch.clamp(x, -128, 127)
    return torch.round(x).to(torch.int8)

def int8_gemm_dequant(a_int8, w_int8, descale, bias, output_dtype):
    a_i32 = a_int8.to(torch.int32)
    w_i32 = w_int8.to(torch.int32)
    y = a_i32 @ w_i32.t()

    y = y + bias
    if output_dtype == torch.float16:
        descale = descale.to(torch.int32)
        descale = descale.view(torch.float32)

    y = y.to(torch.float32) * descale.to(torch.float32)

    return y.to(output_dtype)

def trans_descale_param(fp32_deq_scale):
    uint32__deq_scale = np.frombuffer(fp32_deq_scale, np.uint32)
    uint32__deq_scale &= 0xFFFFE000
    fp32_deq_scale = np.frombuffer(uint32__deq_scale, np.float32)
    uint64_deq_scale = np.zeros(fp32_deq_scale.shape, np.uint64)
    uint64_deq_scale |= np.uint64(uint32__deq_scale)
    uint64_deq_scale |= 1 << 46
    int64_deq_scale = np.int64(uint64_deq_scale)
    int64_deq_scale = torch.from_numpy(int64_deq_scale)
    return int64_deq_scale

def extract_from_nzcache(x, outer_idx, inner_idx, C0_SIZE=16):
    selected = x[outer_idx]
    row_len = x.shape[-1]
    group_num = row_len // C0_SIZE
    positions = [(i * 128 + inner_idx) * C0_SIZE for i in range(group_num)]
    result_chunks = []
    for i in range(group_num):
        start_pos = positions[i]
        row_idx = start_pos // row_len
        col_idx = start_pos % row_len
        chunk = selected[row_idx, 0, col_idx : col_idx + C0_SIZE]
        result_chunks.append(chunk)
    result = torch.cat(result_chunks, dim=0).unsqueeze(0)
    return result

class TestMLAPO(TestCase):

    def gen_random_tensors(
        self,
        N=1,
        headNum=HEADS_Q,
        hiddenDim=HIDDEN,
        dtype=torch.bfloat16,
        cache_mode="krope_ctkv",
        device="npu",
        seed=42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        # hidden
        hidden = (
            torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(N, hiddenDim)))
            .to(dtype)
            .to(device)
        )
        # RMS1
        gamma1 = torch.ones([hiddenDim], dtype=dtype, device=device)
        beta1 = torch.zeros([hiddenDim], dtype=dtype, device=device)

        # MM1 quant
        qscale1 = (
            torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1)))
            .to(dtype)
            .to(device)
        )
        qoffset1 = (
            torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(1)))
            .to(torch.int8)
            .to(device)
        )

        # MM1
        wdqkv = (
            torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(MM1_OUT, hiddenDim)))
            .to(torch.int8)
            .to(device)
        )
        bias1 = (
            torch.from_numpy(np.random.randint(-10, 10, (MM1_OUT)).astype(np.int32))
            .to(torch.int32)
            .to(device)
        )
        descale1 = torch.from_numpy(
            (np.random.rand(MM1_OUT) / 1000).astype(np.float32)
        ).to(device)
        if dtype == torch.float16:
            descale1 = trans_descale_param(np.array(descale1.cpu())).to(device)

        # RMS2 & RMS3
        gamma2 = (
            torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(Q_RMS)))
            .to(dtype)
            .to(device)
        )
        beta2 = (
            torch.from_numpy(np.random.randint(-2, 2, (Q_RMS)).astype(np.float16))
            .to(dtype)
            .to(device)
        )
        gamma3 = (
            torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(K_NOPE)))
            .to(dtype)
            .to(device)
        )

        # MM2 quant
        qscale2 = (
            torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1)))
            .to(dtype)
            .to(device)
        )
        qoffset2 = (
            torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(1)))
            .to(torch.int8)
            .to(device)
        )

        # MM2
        wuq = (
            torch.from_numpy(
                np.random.uniform(-2.0, 2.0, size=(headNum * Q_DIM, Q_RMS))
            )
            .to(torch.int8)
            .to(device)
        )
        bias2 = (
            torch.from_numpy(
                np.random.randint(-10, 10, (headNum * Q_DIM)).astype(np.int32)
            )
            .to(torch.int32)
            .to(device)
        )

        descale2 = torch.from_numpy(
            (np.random.rand(headNum * Q_DIM) / 1000).astype(np.float32)
        ).to(device)
        if dtype == torch.float16:
            descale2 = trans_descale_param(np.array(descale2.cpu())).to(device)

        # BMM3
        wuk = (
            torch.from_numpy(
                np.random.uniform(-2.0, 2.0, size=(headNum, Q_NOPE_DIM, Q_NOPE))
            )
            .to(dtype)
            .to(device)
        )

        # Rope sin/cos
        sin = (
            torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, K_PE)))
            .to(dtype)
            .to(device)
        )
        cos = (
            torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, K_PE)))
            .to(dtype)
            .to(device)
        )

        # caches
        blockNum, blockSize = 1078, 128
        slotMapping = (
            torch.from_numpy(
                np.random.choice(blockNum * blockSize, N, replace=False).astype(
                    np.int32
                )
            )
            .to(torch.int32)
            .to(device)
        )
        keyCache_nope = torch.zeros(blockNum, blockSize, 1, K_NOPE).to(dtype).to(device)
        keyCache_rope = torch.zeros(blockNum, blockSize, 1, K_PE).to(dtype).to(device)
        q_nope_out = torch.zeros((N, headNum, 512), dtype=dtype).to(device)
        q_rope_out = torch.zeros((N, headNum, 64), dtype=dtype).to(device)

        if cache_mode == "int8_nzcache":
            keyCache_nope = keyCache_nope.to(torch.int8)
            q_nope_out = q_nope_out.to(torch.int8)

        ctkv_scale = (
            torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1)))
            .to(dtype)
            .to(device)
        )
        qnope_scale = (
            torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(headNum)))
            .to(dtype)
            .to(device)
        )

        self.dataDict = dict(
            hidden=hidden,
            gamma1=gamma1,
            beta1=beta1,
            qscale1=qscale1,
            qoffset1=qoffset1,
            wdqkv=wdqkv,
            bias1=bias1,
            descale1=descale1,
            gamma2=gamma2,
            beta2=beta2,
            gamma3=gamma3,
            qscale2=qscale2,
            qoffset2=qoffset2,
            wuq=wuq,
            bias2=bias2,
            descale2=descale2,
            wuk=wuk,
            sin=sin,
            cos=cos,
            q_nope_out=q_nope_out,
            q_rope_out=q_rope_out,
            keyCache_nope=keyCache_nope,
            keyCache_rope=keyCache_rope,
            slotMapping=slotMapping,
            ctkv_scale=ctkv_scale,
            qnope_scale=qnope_scale,
        )

    def golden1_torch_npu(
        self,
        N=1,
        headNum=HEADS_Q,
        dtype=torch.bfloat16,
        cache_mode="krope_ctkv",
        dev="npu",
    ):
        # MM1: quantize & quant matmul
        quantOut = quant_per_tensor(
            self.dataDict["hidden"], self.dataDict["qscale1"], self.dataDict["qoffset1"]
        )
        fused = torch_npu.npu_quant_matmul(
            quantOut,
            self.dataDict["wdqkv"].transpose(0, 1),
            self.dataDict["descale1"],
            bias=self.dataDict["bias1"],
            output_dtype=dtype,
        )

        latent, q = fused.split([K_NOPE + K_PE, Q_RMS], dim=-1)
        k_nope = latent[..., :K_NOPE]
        k_pe = latent[..., K_NOPE:].unsqueeze(1)  # [N,1,64]

        # RMSNorm2+3
        q = (
            torch_npu.npu_rms_norm(q.float(), self.dataDict["gamma2"].float(), 1e-6)[0]
            + self.dataDict["beta2"].float()
        )
        k_nope = torch_npu.npu_rms_norm(
            k_nope.float(), self.dataDict["gamma3"].float(), 1e-6
        )[0]
        k_nope = k_nope.unsqueeze(1)  # [N,1,512]

        # MM2: quantize & quant matmul
        quantOut2 = quant_per_tensor(
            q, self.dataDict["qscale2"], self.dataDict["qoffset2"]
        )
        q_out = torch_npu.npu_quant_matmul(
            quantOut2,
            self.dataDict["wuq"].transpose(0, 1),
            self.dataDict["descale2"],
            bias=self.dataDict["bias2"],
            output_dtype=dtype,
        )
        q_out = q_out.view(-1, headNum, Q_DIM)

        q_nope, q_pe = q_out.split(
            [Q_NOPE_DIM, Q_PE_DIM], dim=-1
        )  # [N,16,128], [N,16,64]

        # === BMM3 ===
        q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.dataDict["wuk"]).transpose(
            0, 1
        )
        # === ROPE ===
        q_pe_rope = apply_rope_half(
            q_pe, self.dataDict["cos"], self.dataDict["sin"]
        ).to(dtype)
        k_pe_rope = apply_rope_half(
            k_pe, self.dataDict["cos"], self.dataDict["sin"]
        ).to(dtype)

        if cache_mode == "int8_nzcache":
            q_nope_out = quant_per_tensor_muls(
                q_nope_out,
                self.dataDict["qnope_scale"].reshape(1, headNum, 1),
                torch.zeros_like(q_nope_out),
            )
            k_nope = quant_per_tensor(
                k_nope, self.dataDict["ctkv_scale"], torch.zeros_like(k_nope)
            )
        else:
            q_nope_out = q_nope_out.to(dtype)
            k_nope = k_nope.to(dtype)

        outputs = dict(
            q_nope=q_nope_out,  # [N,16,512]
            q_pe=q_pe_rope,  # [N,16,64]
            k_nope=k_nope,  # [N,1,512]
            k_pe=k_pe_rope,  # [N,1,64]
        )
        return outputs

    def golden2_pytorch(
        self,
        N=1,
        headNum=HEADS_Q,
        dtype=torch.bfloat16,
        cache_mode="krope_ctkv",
        dev="cpu",
    ):
        # MM1: quantize & quant matmul
        quantOut = quant_per_tensor(
            self.dataDict["hidden"], self.dataDict["qscale1"], self.dataDict["qoffset1"]
        )
        fused = int8_gemm_dequant(
            quantOut.cpu(),
            self.dataDict["wdqkv"].cpu(),
            self.dataDict["descale1"].cpu(),
            self.dataDict["bias1"].cpu(),
            output_dtype=dtype,
        ).to(dev)

        latent, q = fused.split([K_NOPE + K_PE, Q_RMS], dim=-1)
        k_nope = latent[..., :K_NOPE]
        k_pe = latent[..., K_NOPE:].unsqueeze(1)  # [N,1,64]

        # RMSNorm2+3
        q = rms_norm(q, self.dataDict["gamma2"]) + self.dataDict["beta2"]
        k_nope = rms_norm(k_nope, self.dataDict["gamma3"]).unsqueeze(1)

        # MM2: quantize & quant matmul
        quantOut2 = quant_per_tensor(
            q, self.dataDict["qscale2"], self.dataDict["qoffset2"]
        )
        q_out = int8_gemm_dequant(
            quantOut2.cpu(),
            self.dataDict["wuq"].cpu(),
            self.dataDict["descale2"].cpu(),
            self.dataDict["bias2"].cpu(),
            output_dtype=dtype,
        ).to(dev)
        q_out = q_out.view(-1, headNum, Q_DIM)

        q_nope, q_pe = q_out.split(
            [Q_NOPE_DIM, Q_PE_DIM], dim=-1
        )  # [N,16,128], [N,16,64]

        # === BMM3 ===
        q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.dataDict["wuk"]).transpose(
            0, 1
        )

        # === ROPE ===
        q_pe_rope = apply_rope_half(
            q_pe, self.dataDict["cos"], self.dataDict["sin"]
        ).to(dtype)
        k_pe_rope = apply_rope_half(
            k_pe, self.dataDict["cos"], self.dataDict["sin"]
        ).to(dtype)

        if cache_mode == "int8_nzcache":
            q_nope_out = quant_per_tensor_muls(
                q_nope_out,
                self.dataDict["qnope_scale"].reshape(1, headNum, 1),
                torch.zeros_like(q_nope_out),
            )
            k_nope = quant_per_tensor(
                k_nope, self.dataDict["ctkv_scale"], torch.zeros_like(k_nope)
            )
        else:
            q_nope_out = q_nope_out.to(dtype)
            k_nope = k_nope.to(dtype)

        outputs = dict(
            q_nope=q_nope_out,  # [N,16,512]
            q_pe=q_pe_rope,  # [N,16,64]
            k_nope=k_nope,  # [N,1,512]
            k_pe=k_pe_rope,  # [N,1,64]
        )
        return outputs

    # (N, headNum, hiddenDim)
    param_combinations = [
        (1, 32, 7168),
        (1, 64, 7168),
        (1, 128, 7168),
        (16, 32, 7168),
        (16, 64, 7168),
        (16, 128, 7168),
        (31, 32, 7168),
        (31, 64, 7168),
        (31, 128, 7168),
        (31, 128, 6144),
    ]

    class GoldenType(IntEnum):
        NPU_SMALL_OPS = 1
        PYTORCH_NATIVE = 2

    cache_mode_names = {1: "krope_ctkv", 2: "int8_nzcache", 3: "nzcache"}

    def run_tests_and_compare(self, cacheMode, golden, dtype, seed=SEED):
        cache_mode = self.cache_mode_names[cacheMode]
        device = "npu"
        if dtype == torch.float16 and cache_mode != "krope_ctkv":
            print("Unsupported combination of dtype and cacheMode!")
            return

        for N, headNum, hiddenDim in self.param_combinations:
            print(
                f"\n=== Testing cache_mode={cache_mode}, N={N}, heads={headNum}, hiddenDim={hiddenDim}, dtype={dtype}, golden={golden}, seed={seed} ==="
            )

            self.gen_random_tensors(
                N=N,
                headNum=headNum,
                hiddenDim=hiddenDim,
                dtype=dtype,
                cache_mode=cache_mode,
                device=device,
                seed=seed,
            )

            if golden == 1:
                golden_out = self.golden1_torch_npu(
                    N=N, headNum=headNum, dtype=dtype, cache_mode=cache_mode, dev=device
                )
            else:
                golden_out = self.golden2_pytorch(
                    N=N, headNum=headNum, dtype=dtype, cache_mode=cache_mode, dev=device
                )

            self.dataDict["wdqkv"] = transdata(self.dataDict["wdqkv"], (16, 32))
            self.dataDict["wdqkv"] = torch_npu.npu_format_cast(
                self.dataDict["wdqkv"], 29
            ).contiguous()
            self.dataDict["wuq"] = transdata(self.dataDict["wuq"], (16, 32))
            self.dataDict["wuq"] = torch_npu.npu_format_cast(
                self.dataDict["wuq"], 29
            ).contiguous()

            blockNum, blockSize = 1078, 128
            torch.ops._C_ascend.mla_preprocess(
                self.dataDict["hidden"].npu(),
                self.dataDict["wdqkv"].npu(),
                self.dataDict["descale1"].npu(),
                self.dataDict["gamma2"].npu(),
                self.dataDict["beta2"].npu(),
                self.dataDict["wuq"].npu(),
                self.dataDict["descale2"].npu(),
                self.dataDict["gamma3"].npu(),
                self.dataDict["cos"].npu(),
                self.dataDict["sin"].npu(),
                self.dataDict["wuk"].npu(),
                self.dataDict["keyCache_nope"].npu(),
                self.dataDict["keyCache_rope"].npu(),
                self.dataDict["slotMapping"].npu(),
                quant_scale0=self.dataDict["qscale1"].npu(),
                quant_offset0=self.dataDict["qoffset1"].npu(),
                bias0=self.dataDict["bias1"].npu(),
                quant_scale1=self.dataDict["qscale2"].npu(),
                quant_offset1=self.dataDict["qoffset2"].npu(),
                bias1=self.dataDict["bias2"].npu(),
                ctkv_scale=self.dataDict["ctkv_scale"].npu(),
                q_nope_scale=self.dataDict["qnope_scale"].npu(),
                cache_mode=cache_mode,
                quant_mode="per_tensor_quant_asymm",
                q_out0=self.dataDict["q_nope_out"].npu(),
                kv_cache_out0=self.dataDict["keyCache_nope"].npu(),
                q_out1=self.dataDict["q_rope_out"].npu(),
                kv_cache_out1=self.dataDict["keyCache_rope"].npu(),
            )

            extracted_k_nope = torch.zeros(N, 1, K_NOPE, dtype=dtype)
            extracted_k_rope = torch.zeros(N, 1, K_PE, dtype=dtype)
            if cache_mode == "int8_nzcache":
                extracted_k_nope = extracted_k_nope.to(torch.int8)
            slotMapping = self.dataDict["slotMapping"].cpu().numpy()
            for i in range(N):
                slot = slotMapping[i]
                outer_idx = slot // blockSize
                inner_idx = slot % blockSize
                if cache_mode == "nzcache":
                    extracted_k_nope[i] = extract_from_nzcache(
                        self.dataDict["keyCache_nope"], outer_idx, inner_idx, C0_SIZE=16
                    )
                    extracted_k_rope[i] = extract_from_nzcache(
                        self.dataDict["keyCache_rope"], outer_idx, inner_idx, C0_SIZE=16
                    )
                elif cache_mode == "int8_nzcache":
                    extracted_k_nope[i] = extract_from_nzcache(
                        self.dataDict["keyCache_nope"], outer_idx, inner_idx, C0_SIZE=32
                    )
                    extracted_k_rope[i] = extract_from_nzcache(
                        self.dataDict["keyCache_rope"], outer_idx, inner_idx, C0_SIZE=16
                    )
                else:
                    extracted_k_nope[i] = self.dataDict["keyCache_nope"][
                        outer_idx, inner_idx
                    ].cpu()
                    extracted_k_rope[i] = self.dataDict["keyCache_rope"][
                        outer_idx, inner_idx
                    ].cpu()

            print(
                "golden q_nope: ",
                golden_out["q_nope"].shape,
                golden_out["q_nope"].sum(),
            )
            print("golden q_rope: ", golden_out["q_pe"].shape, golden_out["q_pe"].sum())
            print(
                "golden keyCache_nope: ",
                golden_out["k_nope"].shape,
                golden_out["k_nope"].sum(),
            )
            print(
                "golden keyCache_rope: ",
                golden_out["k_pe"].shape,
                golden_out["k_pe"].sum(),
            )
            print(" ================================================== ")
            print(
                "q_nope_out: ",
                self.dataDict["q_nope_out"].shape,
                self.dataDict["q_nope_out"].sum(),
            )
            print(
                "q_rope_out: ",
                self.dataDict["q_rope_out"].shape,
                self.dataDict["q_rope_out"].sum(),
            )
            print(
                "extracted keyCache_nope: ",
                extracted_k_nope.shape,
                extracted_k_nope.sum(),
            )
            print(
                "extracted keyCache_rope: ",
                extracted_k_rope.shape,
                extracted_k_rope.sum(),
            )

            print(" ========== compare results: ========== ")
            print(
                " q_nope diff: ",
                torch.allclose(
                    golden_out["q_nope"].cpu(),
                    self.dataDict["q_nope_out"].cpu(),
                    atol=0.001,
                    rtol=0.001,
                ),
            )
            print(
                " q_nope max diff: ",
                torch.max(
                    torch.abs(
                        golden_out["q_nope"].cpu() - self.dataDict["q_nope_out"].cpu()
                    )
                ),
            )
            print(
                " q_rope diff: ",
                torch.allclose(
                    golden_out["q_pe"].cpu(),
                    self.dataDict["q_rope_out"].cpu(),
                    atol=0.001,
                    rtol=0.001,
                ),
            )
            print(
                " q_rope max diff: ",
                torch.max(
                    torch.abs(
                        golden_out["q_pe"].cpu() - self.dataDict["q_rope_out"].cpu()
                    )
                ),
            )

            print(
                " keyCache_nope diff: ",
                torch.allclose(
                    golden_out["k_nope"].cpu(),
                    extracted_k_nope,
                    atol=0.001,
                    rtol=0.001,
                ),
            )
            print(
                " keyCache_nope max diff: ",
                torch.max(torch.abs(golden_out["k_nope"].cpu() - extracted_k_nope)),
            )
            print(
                " keyCache_rope diff: ",
                torch.allclose(
                    golden_out["k_pe"].cpu(),
                    extracted_k_rope,
                    atol=0.001,
                    rtol=0.001,
                ),
            )
            print(
                " keyCache_rope max diff: ",
                torch.max(torch.abs(golden_out["k_pe"].cpu() - extracted_k_rope)),
            )

            self.assertTrue(
                torch.allclose(
                    golden_out["q_nope"].cpu(),
                    self.dataDict["q_nope_out"].cpu(),
                    atol=0.001,
                    rtol=0.001,
                ),
                f"q_nope mismatch. max diff: {torch.max(torch.abs(golden_out['q_nope'].cpu() - self.dataDict['q_nope_out'].cpu()))}",
            )

            self.assertTrue(
                torch.allclose(
                    golden_out["q_pe"].cpu(),
                    self.dataDict["q_rope_out"].cpu(),
                    atol=0.001,
                    rtol=0.001,
                ),
                f"q_rope mismatch. max diff: {torch.max(torch.abs(golden_out['q_pe'].cpu() - self.dataDict['q_rope_out'].cpu()))}",
            )

            self.assertTrue(
                torch.allclose(
                    golden_out["k_nope"].cpu(), extracted_k_nope, atol=0.001, rtol=0.001
                ),
                f"keyCache_nope mismatch. max diff: {torch.max(torch.abs(golden_out['k_nope'].cpu() - extracted_k_nope))}",
            )

            self.assertTrue(
                torch.allclose(
                    golden_out["k_pe"].cpu(), extracted_k_rope, atol=0.001, rtol=0.001
                ),
                f"keyCache_rope mismatch. max diff: {torch.max(torch.abs(golden_out['k_pe'].cpu() - extracted_k_rope))}",
            )

    def test_mla_preprocess_ops_bf16_cachemode1_golden1(self):
        self.run_tests_and_compare(
            cacheMode=1,
            golden=self.GoldenType.NPU_SMALL_OPS,
            dtype=torch.bfloat16,
            seed=SEED,
        )

    def test_mla_preprocess_ops_bf16_cachemode2_golden1(self):
        self.run_tests_and_compare(
            cacheMode=2,
            golden=self.GoldenType.NPU_SMALL_OPS,
            dtype=torch.bfloat16,
            seed=SEED,
        )

    def test_mla_preprocess_ops_bf16_cachemode3_golden1(self):
        self.run_tests_and_compare(
            cacheMode=3,
            golden=self.GoldenType.NPU_SMALL_OPS,
            dtype=torch.bfloat16,
            seed=SEED,
        )

    def test_mla_preprocess_ops_fp16_cachemode1_golden1(self):
        self.run_tests_and_compare(
            cacheMode=1,
            golden=self.GoldenType.NPU_SMALL_OPS,
            dtype=torch.float16,
            seed=SEED,
        )

    def test_mla_preprocess_ops_bf16_cachemode1_golden2(self):
        self.run_tests_and_compare(
            cacheMode=1,
            golden=self.GoldenType.PYTORCH_NATIVE,
            dtype=torch.bfloat16,
            seed=SEED,
        )

    def test_mla_preprocess_ops_bf16_cachemode2_golden2(self):
        self.run_tests_and_compare(
            cacheMode=2,
            golden=self.GoldenType.PYTORCH_NATIVE,
            dtype=torch.bfloat16,
            seed=SEED,
        )

    def test_mla_preprocess_ops_bf16_cachemode3_golden2(self):
        self.run_tests_and_compare(
            cacheMode=3,
            golden=self.GoldenType.PYTORCH_NATIVE,
            dtype=torch.bfloat16,
            seed=SEED,
        )

    def test_mla_preprocess_ops_fp16_cachemode1_golden2(self):
        self.run_tests_and_compare(
            cacheMode=1,
            golden=self.GoldenType.PYTORCH_NATIVE,
            dtype=torch.float16,
            seed=SEED,
        )

if __name__ == "__main__":
    run_tests()
