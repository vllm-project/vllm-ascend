"""
replace_function.py
function: replace a function in a Python file with new source code.
usage: python replace_function.py
"""

import ast
import textwrap

TARGET_FILE = "/usr/local/python3.11.14/lib/python3.11/site-packages/modelscope/utils/hf_util/patcher.py"

FUNCTION_NAME = "get_model_dir"

NEW_FUNCTION_SOURCE = """
def get_model_dir(pretrained_model_name_or_path,
                    ignore_file_pattern=None,
                    allow_file_pattern=None,
                    **kwargs):
    # This is the func after patched
    from modelscope import snapshot_download
    subfolder = kwargs.pop('subfolder', None)
    file_filter = None
    if subfolder:
        file_filter = f'{subfolder}/*'
    if not os.path.exists(pretrained_model_name_or_path):
        revision = kwargs.pop('revision', None)
        if revision is None or revision == 'main':
            revision = 'master'
        if file_filter is not None:
            allow_file_pattern = file_filter
        local_files_only = kwargs.pop('local_files_only', False)
        model_dir = snapshot_download(
            pretrained_model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            ignore_file_pattern=ignore_file_pattern,
            allow_file_pattern=allow_file_pattern)
        if subfolder:
            model_dir = os.path.join(model_dir, subfolder)
    else:
        model_dir = pretrained_model_name_or_path
    return model_dir
"""


def replace_function_in_file(filepath: str, func_name: str, new_source: str) -> None:
    with open(filepath, encoding="utf-8") as f:
        original_source = f.read()

    lines = original_source.splitlines(keepends=True)

    tree = ast.parse(original_source, filename=filepath)

    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                target_node = node
                break

    if target_node is None:
        raise ValueError(f"function '{func_name}' not found in: {filepath}")

    start_line = target_node.lineno
    end_line = target_node.end_lineno

    print(f"[info] find '{func_name}' in {start_line}–{end_line}")

    def_line = lines[start_line - 1]
    indent = len(def_line) - len(def_line.lstrip())
    indent_str = def_line[:indent]

    new_dedented = textwrap.dedent(new_source).strip("\n")
    new_indented = textwrap.indent(new_dedented, indent_str) + "\n"

    backup_path = filepath + ".bak"
    with open(backup_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[info] origin module backup to: {backup_path}")

    new_lines = lines[: start_line - 1] + [new_indented] + lines[end_line:]

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"[ok] function '{func_name}' replace success: {filepath}")


def verify_syntax(filepath: str) -> None:
    """ast check the modified file to ensure no syntax errors."""
    with open(filepath, encoding="utf-8") as f:
        source = f.read()
    try:
        ast.parse(source, filename=filepath)
        print("[ok] check passed, no syntax errors detected.")
    except SyntaxError as e:
        print(f"[error] syntax error: {e}")
        raise


if __name__ == "__main__":
    replace_function_in_file(TARGET_FILE, FUNCTION_NAME, NEW_FUNCTION_SOURCE)
    verify_syntax(TARGET_FILE)
