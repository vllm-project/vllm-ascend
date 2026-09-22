#!/usr/bin/env python3
"""Bundle the active CANN package with its matching FlashAttn native ABI.

Only FlashAttn's official wrapper/source is replaced. Other CANN operators keep
the package and builder selected by the build environment, including legacy
flat packages. The extension is compiled here, never at service startup.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_flash_attn(root, *, compile_extension=True):
    spec = importlib.util.find_spec("cann_ops_transformer")
    if spec is None or spec.origin is None:
        raise RuntimeError("The selected CANN environment has no cann_ops_transformer package")
    original = Path(spec.origin).resolve().parent
    generated = root / "vllm_ascend/_cann_ops_custom/python/cann_ops_transformer"
    if original == generated.resolve():
        raise RuntimeError("Build against the original CANN package, not a previous VA artifact")
    if generated.exists():
        shutil.rmtree(generated)
    shutil.copytree(original, generated, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    distributed = (original / "ops/attention/flash_attn/flash_attn.py").is_file()
    wrapper_rel = Path("ops/attention/flash_attn/flash_attn.py" if distributed else "ops/flash_attn.py")
    cpp_rel = Path("csrc/attention/flash_attn.cpp" if distributed else "ops/csrc/flash_attn.cpp")
    if not (original / wrapper_rel).is_file():
        raise RuntimeError(f"Unsupported official FlashAttn package structure: {original}")

    official = root / "csrc/attention/flash_attn/torch_extension"
    wrapper = (official / "flash_attn.py").read_text()
    # Keep the official builder for every other operator. Legacy CANN exposes
    # AS_LIBRARY and registers schemas in __init__, rather than lazily.
    if not distributed:
        wrapper = wrapper.replace(
            "from cann_ops_transformer.op_builder import OpBuilder, get_as_library",
            "from cann_ops_transformer.op_builder.builder import OpBuilder, AS_LIBRARY",
        )
        wrapper = wrapper.replace('("flash_attn", category="attention")', '("flash_attn")')
        wrapper = wrapper.replace("flash_attn_op_builder._ensure_initialized()", "")
        wrapper = wrapper.replace("get_as_library()", "AS_LIBRARY")
    # The official torch namespace/API is unchanged. Only the build-time versus
    # runtime extension loader differs from the upstream JIT wrapper.
    anchor = "    def sources(self):"
    assert wrapper.count(anchor) == 1
    wrapper = wrapper.replace(
        anchor,
        """    def load(self, verbose=True):
        from cann_ops_transformer._C import flash_attn

        return flash_attn

"""
        + anchor,
    )
    wrapper = wrapper.replace('return ["csrc/attention/flash_attn.cpp"]', f'return ["{cpp_rel.as_posix()}"]')
    # Match the native no-LSE contract; GQA without DCP uses this branch.
    anchor = "            return (\n                torch.empty(attention_out_size"
    assert wrapper.count(anchor) == 1
    wrapper = wrapper.replace(
        anchor, "            if not return_softmax_lse:\n                softmax_out_size = (0,)\n\n" + anchor
    )
    (generated / wrapper_rel).write_text(wrapper)
    (generated / cpp_rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(official / "csrc/flash_attn.cpp", generated / cpp_rel)
    native = generated / "_C"
    native.mkdir(exist_ok=True)
    (native / "__init__.py").write_text("")

    modified = {wrapper_rel.as_posix(), cpp_rel.as_posix()}
    originals = {}
    for path in original.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel = path.relative_to(original).as_posix()
        originals[rel] = sha256(path)
        if rel not in modified and sha256(generated / rel) != originals[rel]:
            raise RuntimeError(f"An unrelated official module changed: {rel}")

    if compile_extension:
        import torch
        import torch_npu
        from torch.utils.cpp_extension import load

        cann = Path(os.environ["ASCEND_HOME_PATH"])
        torch_npu_path = Path(torch_npu.__file__).parent
        vendor = root / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
        includes = [
            vendor / "op_api/include",
            torch_npu_path / "include",
            torch_npu_path / "include/third_party/hccl/inc",
            torch_npu_path / "include/third_party/acl/inc",
            torch_npu_path / "include/third_party/op-plugin",
            cann / "include",
            cann / "include/aclnnop",
            generated / "common",
            generated / "common/inc",
        ]
        build_dir = root / "build/flash_attn_binding"
        build_dir.mkdir(parents=True, exist_ok=True)
        module = load(
            name="flash_attn",
            sources=[str(generated / cpp_rel)],
            extra_include_paths=[str(path) for path in includes if path.is_dir()],
            extra_cflags=["-O3", "-fvisibility=hidden"],
            extra_ldflags=[f"-L{cann / 'lib64'}", "-lascendcl", f"-L{torch_npu_path / 'lib'}", "-ltorch_npu"],
            build_directory=str(build_dir),
            verbose=True,
        )
        shutil.copy2(module.__file__, native / "flash_attn.so")
        torch_version = torch.__version__
    else:
        torch_version = None

    manifest = {
        "cann_home": os.environ.get("ASCEND_HOME_PATH"),
        "original_package": str(original),
        "distributed_package": distributed,
        "torch_version": torch_version,
        "original_files": originals,
        "modified_files": sorted(modified),
        "generated_files": {str(p.relative_to(generated)): sha256(p) for p in generated.rglob("*") if p.is_file()},
    }
    (generated.parent / "flash_attn_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "package": str(generated),
                "layout": "distributed" if distributed else "flat",
                "preserved_files": len(originals) - len(modified),
                "compiled": compile_extension,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-only", action="store_true", help="Check package structure without building")
    args = parser.parse_args()
    package_flash_attn(args.root.resolve(), compile_extension=not args.source_only)
