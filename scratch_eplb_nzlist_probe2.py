"""
Probe v2 for W4A8MXFP EPLB on a device where clone/copy_ reject internal (NZ) format.
RUN ON THE REAL DEPLOYMENT DEVICE.

Since clone/copy_ can't touch NZ, the only remaining hope for an in-place, data_ptr-
preserving NZ writeback (needed for aclgraph) is an in-place format cast. This probe:

  T1  build a STANDALONE NZ tensor via format_cast from standalone ND 2D (no clone).
  T2  copy_ standalone NZ <- standalone NZ (both format_cast-built).
  T3  copy_ standalone NZ <- ND.
  T4  in-place format cast: does it exist, and does it PRESERVE data_ptr?
      (this is the graph-safe writeback we need)
  T5  format_cast(..., out=existing) style: can we target existing storage?
"""

import torch
import torch_npu

NZ = 29
ND = 2
DEV = "npu:0"
TWO_I, H_HALF = 4096, 2048  # one expert: (2I, H//2) uint8 packed


def fmt(t):
    try:
        return f"format={torch_npu.get_npu_format(t)} offset={t.storage_offset()} " \
               f"contig={t.is_contiguous()} shape={tuple(t.shape)} ptr={t.data_ptr()}"
    except Exception as e:
        return f"<fmt err {e}>"


def make_expert_nz():
    """Standalone NZ expert built WITHOUT clone: fresh ND uint8 -> format_cast NZ."""
    nd = torch.randint(0, 255, (TWO_I, H_HALF), dtype=torch.uint8, device=DEV)
    nz = torch_npu.npu_format_cast(
        nd, NZ, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
    )
    return nd, nz


def main():
    torch.npu.set_device(0)

    print("==== T1: format_cast standalone ND(2D) -> NZ (build without clone) ====")
    try:
        nd, nz = make_expert_nz()
        print("  nd:", fmt(nd))
        print("  nz:", fmt(nz))
    except Exception as e:
        print("  T1 FAIL:", repr(e)[:200])
        return

    print("\n==== T2: copy_ standalone NZ <- standalone NZ ====")
    try:
        _, dst = make_expert_nz()
        _, src = make_expert_nz()
        p0 = dst.data_ptr()
        dst.copy_(src)
        print("  OK, ptr_preserved=", dst.data_ptr() == p0)
    except Exception as e:
        print("  T2 FAIL:", repr(e)[:200])

    print("\n==== T3: copy_ standalone NZ <- ND ====")
    try:
        _, dst = make_expert_nz()
        nd_buf = torch.randint(0, 255, (TWO_I, H_HALF), dtype=torch.uint8, device=DEV)
        dst.copy_(nd_buf)
        print("  OK")
    except Exception as e:
        print("  T3 FAIL:", repr(e)[:200])

    print("\n==== T4: in-place format cast, data_ptr preserved? ====")
    # try both a tensor method and the module-level fn, ND->NZ and NZ->NZ
    nd, nz = make_expert_nz()
    print("  has Tensor.npu_format_cast_:", hasattr(nd, "npu_format_cast_"))
    print("  has torch_npu.npu_format_cast_:", hasattr(torch_npu, "npu_format_cast_"))
    # (a) ND tensor -> NZ in place
    try:
        nd2 = torch.randint(0, 255, (TWO_I, H_HALF), dtype=torch.uint8, device=DEV)
        p0 = nd2.data_ptr()
        nd2.npu_format_cast_(NZ)
        print("  (a) nd.npu_format_cast_(NZ): OK  format=", torch_npu.get_npu_format(nd2),
              " ptr_preserved=", nd2.data_ptr() == p0)
    except Exception as e:
        print("  (a) nd.npu_format_cast_(NZ) FAIL:", repr(e)[:160])
    # (b) module-level, src ND into dst-format
    try:
        nd3 = torch.randint(0, 255, (TWO_I, H_HALF), dtype=torch.uint8, device=DEV)
        p0 = nd3.data_ptr()
        torch_npu.npu_format_cast_(nd3, nz)   # cast nd3 to nz's format, in place?
        print("  (b) torch_npu.npu_format_cast_(nd, nz): OK format=",
              torch_npu.get_npu_format(nd3), " ptr_preserved=", nd3.data_ptr() == p0)
    except Exception as e:
        print("  (b) torch_npu.npu_format_cast_(nd,nz) FAIL:", repr(e)[:160])

    print("\n==== T5: format_cast into existing storage (out=) ? ====")
    try:
        nd_src = torch.randint(0, 255, (TWO_I, H_HALF), dtype=torch.uint8, device=DEV)
        _, dst_nz = make_expert_nz()
        p0 = dst_nz.data_ptr()
        # some builds accept out=
        torch_npu.npu_format_cast(nd_src, NZ, out=dst_nz,
                                  customize_dtype=torch.float8_e4m3fn,
                                  input_dtype=torch_npu.float4_e2m1fn_x2)
        print("  out= OK, ptr_preserved=", dst_nz.data_ptr() == p0)
    except Exception as e:
        print("  out= FAIL:", repr(e)[:160])

    print("\n[done]")


if __name__ == "__main__":
    main()
