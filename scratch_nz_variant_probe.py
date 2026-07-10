"""
Why does copy_/clone work on W8A8's NZ weight but fail on W4A8MXFP's NZ weight,
even though both are "NZ"?  Hypothesis: they are DIFFERENT NZ variants.
  * W8A8      : plain FRACTAL_NZ        (npu_format_cast(x, 29), no customize dtypes)
  * W4A8MXFP  : FRACTAL_NZ_C0_16        (npu_format_cast(x, 29, customize_dtype=fp8, input_dtype=fp4))
This script builds both and tests copy_ / clone on each.  RUN ON THE REAL DEVICE.
"""

import torch
import torch_npu

NZ = 29
DEV = "npu:0"
N, K = 2048, 4096


def fmt(t):
    return torch_npu.get_npu_format(t)


def test_variant(title, make):
    print(f"\n==== {title} ====")
    try:
        dst = make()
        print("  format:", fmt(dst), "dtype:", dst.dtype,
              "contig:", dst.is_contiguous(), "offset:", dst.storage_offset())
    except Exception as e:
        print("  build FAIL:", repr(e)[:180])
        return
    # copy_ from another same-variant tensor
    try:
        src = make()
        p0 = dst.data_ptr()
        dst.copy_(src)
        print("  copy_ NZ<-NZ: OK  (ptr_preserved=%s)" % (dst.data_ptr() == p0))
    except Exception as e:
        print("  copy_ NZ<-NZ FAIL:", repr(e)[:180])
    # copy_ from a plain ND tensor
    try:
        nd = torch.randint(0, 100, (N, K), dtype=dst.dtype, device=DEV)
        make().copy_(nd)
        print("  copy_ NZ<-ND: OK")
    except Exception as e:
        print("  copy_ NZ<-ND FAIL:", repr(e)[:180])
    # clone
    try:
        make().clone()
        print("  clone: OK")
    except Exception as e:
        print("  clone FAIL:", repr(e)[:180])


def main():
    torch.npu.set_device(0)

    # W8A8-style: plain FRACTAL_NZ from int8, no customize dtypes
    def make_plain_nz():
        nd = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=DEV)
        return torch_npu.npu_format_cast(nd, NZ)

    # W4A8MXFP-style: FRACTAL_NZ_C0_16 from uint8 with mxfp customize dtypes
    def make_c0_16_nz():
        nd = torch.randint(0, 255, (N, K), dtype=torch.uint8, device=DEV)
        return torch_npu.npu_format_cast(
            nd, NZ, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )

    test_variant("W8A8-style  plain FRACTAL_NZ (int8)", make_plain_nz)
    test_variant("W4A8MXFP-style  FRACTAL_NZ_C0_16 (uint8, fp8/fp4)", make_c0_16_nz)

    print("\n[done]")


if __name__ == "__main__":
    main()
