from argparse import ArgumentParser

import libcst as cst
import libcst.matchers as m

# Patch the benchmark_dataset.py file to set streaming=False in load_dataset calls


# TDOO(Potabk): Remove this patch when the issue is fixed in the upstream
class StreamingFalseTransformer(cst.CSTTransformer):

    def __init__(self):
        self.in_target_class = False
        self.in_target_func = False

    def visit_ClassDef(self, node):
        if node.name.value == "HuggingFaceDataset":
            self.in_target_class = True

    def leave_ClassDef(self, original_node, updated_node):
        self.in_target_class = False
        return updated_node

    def visit_FunctionDef(self, node):
        if self.in_target_class and node.name.value == "load_data":
            self.in_target_func = True

    def leave_FunctionDef(self, original_node, updated_node):
        self.in_target_func = False
        return updated_node

    def leave_Call(self, original_node, updated_node):
        if self.in_target_class and self.in_target_func:
            if m.matches(updated_node.func, m.Name("load_dataset")):
                new_args = []
                for arg in updated_node.args:
                    if arg.keyword and arg.keyword.value == "streaming":
                        new_arg = arg.with_changes(value=cst.Name("False"))
                        new_args.append(new_arg)
                    else:
                        new_args.append(arg)
                return updated_node.with_changes(args=new_args)
        return updated_node


def patch_file(path):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    module = cst.parse_module(source)
    modified = module.visit(StreamingFalseTransformer())

    with open(path, "w", encoding="utf-8") as f:
        f.write(modified.code)

    print(f"Patched: {path}")


if __name__ == '__main__':
    parser = ArgumentParser(
        description=
        "Patch benchmark_dataset.py to set streaming=False in load_dataset calls"
    )
    parser.add_argument("--path",
                        type=str,
                        help="Path to the benchmark_dataset.py file")
    args = parser.parse_args()
    patch_file(args.path)
