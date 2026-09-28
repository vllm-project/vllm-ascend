# SPDX-License-Identifier: Apache-2.0

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import regex as re


@pytest.mark.parametrize(
    "folder, op_name",
    [
        ("sparse_flash_attention", "SparseFlashAttention"),
        ("kv_quant_sparse_flash_attention", "KvQuantSparseFlashAttention"),
    ],
)
@pytest.mark.parametrize("options_v2", [False, True])
def test_sfa_prefix_compile_options_are_a3_only(tmp_path, folder, op_name, options_v2):
    root = Path(__file__).resolve().parents[3]
    util = root / "csrc/cmake/scripts/util"
    cmake = (root / "csrc/attention" / folder / "op_host/CMakeLists.txt").read_text()
    config = tmp_path / "custom_compile_options.ini"

    # Exercise the actual operator configuration through both INI formats.
    for block in re.findall(r"add_ops_compile_options\(([^)]*)\)", cmake):
        tokens = re.sub(r"#[^\n]*", "", block).split()
        name = tokens[tokens.index("OP_NAME") + 1]
        options_start = tokens.index("OPTIONS")
        options = tokens[options_start + 1 :]
        socs = tokens[tokens.index("COMPUTE_UNIT") + 1 : options_start] if "COMPUTE_UNIT" in tokens else []
        if options_v2:
            subprocess.run(
                [sys.executable, str(util / "ascendc_gen_options.py"), str(config), name, *socs, *options],
                check=True,
            )
        else:
            with config.open("a") as stream:
                stream.write(f"{name},{';'.join(socs)},{';'.join(options)}\n")

    parser = runpy.run_path(str(util / "opdesc_parser.py"))
    op = parser["OpDesc"](op_name)
    parser["_get_op_custom_options"]([op], str(tmp_path))
    macro = "-DVLLM_ASCEND_SFA_A3"
    assert op.custom_compile_options["ascend910_93"] == [macro]
    assert all(macro not in options for soc, options in op.custom_compile_options.items() if soc != "ascend910_93")
