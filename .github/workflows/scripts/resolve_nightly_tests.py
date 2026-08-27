import argparse
import base64
import json
import os
import subprocess

try:
    import yaml
except ImportError:
    subprocess.check_call(["pip3", "install", "pyyaml", "-q"])
    import yaml


def _collect_names(node, names):
    """Recursively collect `name` fields from list-of-dict entries."""
    if isinstance(node, dict):
        for v in node.values():
            _collect_names(v, names)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.add(item["name"])
            elif isinstance(item, (dict, list)):
                _collect_names(item, names)


def parse_nightly_matrix(b64_content, soc):
    """Read base64-encoded matrix YAML and return a set of all `name` strings under `soc`."""
    if not b64_content:
        return set()
    try:
        parsed = yaml.safe_load(base64.b64decode(b64_content).decode())
        names = set()
        soc_block = parsed.get(soc, {})
        if isinstance(soc_block, dict):
            _collect_names(soc_block, names)
        return names
    except Exception:
        return set()


def _walk(matrix, path):
    """Walk a dot-separated path in the matrix; return the node at the end, or {} if missing."""
    node = matrix
    for key in path.split("."):
        if isinstance(node, dict):
            node = node.get(key, {})
        else:
            return {}
    return node


def cmd_dispatch(_args):
    """Resolve /nightly <name> tokens and emit dispatch flags.

    Reads NIGHTLY_MATRIX (base64-encoded nightly_config.yaml) and TEST_CASES
    (comma-separated tokens from the /nightly comment). Writes to GITHUB_OUTPUT:
      - dispatch_a2=true|false
      - dispatch_a3=true|false
      - test_cases=<group,model>  (only when a /<model> token matched an accuracy group)
    """
    # Prefer NIGHTLY_MATRIX for backward compatibility; fall back to WEEKLY_MATRIX
    # so the same script can resolve test names from either config family.
    matrix_b64 = os.environ.get("NIGHTLY_MATRIX", "") or os.environ.get("WEEKLY_MATRIX", "")
    a2_names = parse_nightly_matrix(matrix_b64, "a2")
    a3_names = parse_nightly_matrix(matrix_b64, "a3")

    raw = os.environ.get("TEST_CASES", "")
    test_cases = [tc.strip() for tc in raw.split(",") if tc.strip()]

    da2, da3 = False, False
    transformed = None
    for tc in test_cases:
        if "/" in tc:
            g, m = tc.split("/", 1)
            if g in a2_names:
                da2 = True
                transformed = f"{g},{m}"
            break
        elif tc == "accuracy-group":
            da2 = True
        else:
            if tc in a2_names:
                da2 = True
            if tc in a3_names:
                da3 = True

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        if transformed:
            f.write(f"test_cases={transformed}\n")
        f.write(f"dispatch_a2={str(da2).lower()}\n")
        f.write(f"dispatch_a3={str(da3).lower()}\n")


def cmd_matrix(_args):
    """Extract matrix sections per MATRIX_OUTPUTS spec.

    Reads MATRIX_FILE (path to nightly_config.yaml) and MATRIX_OUTPUTS
    (JSON object {output_name: "dot.path.in.yaml"}). Writes to GITHUB_OUTPUT:
      - <output_name>=<json array>  (one entry per spec item)
    """
    matrix_file = os.environ.get("MATRIX_FILE", "")
    if not matrix_file:
        raise SystemExit("MATRIX_FILE env var is required for --mode=matrix")
    spec = json.loads(os.environ.get("MATRIX_OUTPUTS", "{}"))

    with open(matrix_file) as f:
        matrix = yaml.safe_load(f) or {}

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for name, path in spec.items():
            node = _walk(matrix, path)
            cfg = node if isinstance(node, list) else []
            f.write(f"{name}={json.dumps(cfg)}\n")


_MODES = {
    "dispatch": cmd_dispatch,
    "matrix": cmd_matrix,
}


def main():
    parser = argparse.ArgumentParser(
        description="Nightly matrix config helper (used by /nightly dispatcher and A2/A3 workflows).",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(_MODES.keys()),
        default="dispatch",
        help="Operation mode. 'dispatch' resolves /nightly names into A2/A3 dispatch flags "
        "(used by pr_nightly_command.yml). 'matrix' extracts matrix sections per a JSON "
        "spec (used by A2/A3 generate-* jobs). Default: dispatch.",
    )
    args = parser.parse_args()
    _MODES[args.mode](args)


if __name__ == "__main__":
    main()
