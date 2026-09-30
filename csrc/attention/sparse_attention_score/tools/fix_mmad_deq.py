from pathlib import Path


def fix_mmad(path, bsa_path, anchor_end, deq_sig_suffix):
    bsa = Path(bsa_path).read_text(encoding="utf-8")
    sasa = Path(path).read_text(encoding="utf-8")

    idx = bsa.find(deq_sig_suffix)
    if idx < 0:
        raise SystemExit(f"marker not found in BSA: {deq_sig_suffix}")
    start = bsa.rfind("    template <class TensorB, class TensorC>", 0, idx)
    end = bsa.find("protected:", idx)
    deq_op = bsa[start:end].rstrip() + "\n\n"

    oidx = sasa.find(deq_sig_suffix)
    if oidx >= 0:
        line_start = sasa.rfind("\n", 0, oidx) + 1
        sync = "SetCrossCoreSync<4, PIPE_FIX>"
        pos = oidx
        count = 0
        end_pos = -1
        while True:
            pos = sasa.find(sync, pos)
            if pos < 0:
                break
            count += 1
            if count == 2:
                end_pos = sasa.find("\n    }\n", pos)
                break
            pos += len(sync)
        if end_pos < 0:
            raise SystemExit("end not found")
        end_pos += len("\n    }\n")
        sasa = sasa[:line_start] + sasa[end_pos:]

    insert_at = sasa.find(anchor_end)
    if insert_at < 0:
        raise SystemExit(f"anchor not found: {anchor_end}")
    insert_at += len(anchor_end)
    if deq_sig_suffix not in sasa[insert_at : insert_at + 8000]:
        sasa = sasa[:insert_at] + "\n" + deq_op + sasa[insert_at:]

    while sasa.count("protected:") > 1:
        first = sasa.find("protected:")
        second = sasa.find("protected:", first + 1)
        chunk = sasa[first:second]
        if "copyL1ToL0A" not in chunk:
            sasa = sasa[:first] + sasa[second:]
        else:
            break

    Path(path).write_text(sasa, encoding="utf-8")
    print("fixed", path)


if __name__ == "__main__":
    sasa_root = Path(__file__).resolve().parents[1]
    attn_root = sasa_root.parent
    ops_root = attn_root.parent
    fix_mmad(
        sasa_root / "op_kernel/attn_infra/gemm/block/block_mmad_qk_arch35_ABf16_C_to_UB.hpp",
        attn_root / "block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_qk_arch35_ABf16_C_to_UB.hpp",
        "SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);\n    }",
        "Arch::CrossCoreFlag mm1ToSmFlag, uint64_t deqScalar)",
    )
    fix_mmad(
        sasa_root / "op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp",
        attn_root / "block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp",
        "SetCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);\n    }",
        "Arch::CrossCoreFlag mm2ToReFlag, uint64_t deqScalar)",
    )
