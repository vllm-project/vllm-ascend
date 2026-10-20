# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the DCP dispatcher contracts without importing Torch or the extension."""

import ast
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OPERATORS = (
    "npu_quant_lightning_indexer_v2",
    "npu_quant_lightning_indexer_v3",
    "npu_quant_lightning_indexer_v2_metadata",
    "npu_sparse_attn_sharedkv",
    "npu_sparse_attn_sharedkv_metadata",
)
CPP_TYPES = {
    "at::Tensor": "Tensor",
    "c10::optional<at::Tensor>": "Tensor?",
    "int64_t": "int",
    "double": "float",
    "bool": "bool",
    "c10::string_view": "str",
}


def function_argument_types(source: str, function: str) -> list[str]:
    declaration = re.search(r"\b" + function.split("::")[-1] + r"\s*\((.*?)\)\s*\{", source, re.S)
    arguments = declaration[1].split(",")
    result = []
    for argument in arguments:
        argument = argument.split("=", 1)[0].strip()
        cpp_type = re.sub(r"\b\w+\s*$", "", argument)
        cpp_type = re.sub(r"\bconst\b|\s|&", "", cpp_type)
        result.append(CPP_TYPES[cpp_type])
    return result


def operator_schemas(source: str) -> dict[str, list[str]]:
    result = {}
    for definition in re.finditer(r'ops\.def\(\s*((?:"(?:\\.|[^"\\])*"\s*)+)\)', source):
        schema = "".join(ast.literal_eval(part) for part in re.findall(r'"(?:\\.|[^"\\])*"', definition[1]))
        name, arguments = schema.split("(", 1)
        if name in OPERATORS:
            result[name] = [
                argument.strip().split()[0]
                for argument in arguments.split(")", 1)[0].split(",")
                if argument.strip() != "*"
            ]
    return result


class TestDsv4DcpBindingSchemas(unittest.TestCase):
    def test_private_and_meta_kernels_match_each_schema(self):
        binding = (REPO_ROOT / "csrc/torch_binding.cpp").read_text()
        meta = (REPO_ROOT / "csrc/torch_binding_meta.cpp").read_text()
        candidate = (
            REPO_ROOT / "csrc/attention/quant_lightning_indexer_v2/quant_lightning_indexer_v2_torch_adpt.h"
        ).read_text()
        schemas = operator_schemas(binding)
        self.assertEqual(set(schemas), set(OPERATORS))
        registrations = {}
        for source, implicit_dispatch in ((binding, None), (meta, "Meta")):
            for registration in re.finditer(
                r'ops\.impl\("([^"]+)",\s*(?:torch::k(PrivateUse1|Meta),\s*)?&([\w:]+)\s*\)', source
            ):
                name, dispatch, function = registration.groups()
                if name in OPERATORS:
                    key = (name, dispatch or implicit_dispatch)
                    self.assertNotIn(key, registrations)
                    function_source = candidate if function.startswith("vllm_ascend::qli_v2::") else source
                    registrations[key] = function_argument_types(function_source, function)

        for name in OPERATORS:
            self.assertEqual(schemas[name][-1], "Tensor?", name)
            for dispatch in ("PrivateUse1", "Meta"):
                with self.subTest(operator=name, dispatch=dispatch):
                    self.assertEqual(registrations[(name, dispatch)], schemas[name])


if __name__ == "__main__":
    unittest.main()
