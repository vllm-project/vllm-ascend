import argparse
import base64
import json
import os
import string
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


# Values coming from a PR-derived config are interpolated into downstream
# matrices (runner labels, test paths); restrict them to shell-safe
# characters and require runner labels to exist in the base-branch config.
_SAFE_VALUE_CHARS = frozenset(string.ascii_letters + string.digits + "_./:@+, -")


def _collect_runner_labels(node, labels):
    """Recursively collect runner labels (`os` / `runner` fields)."""
    if isinstance(node, dict):
        for key in ("os", "runner"):
            value = node.get(key)
            if isinstance(value, str):
                labels.add(value)
        for value in node.values():
            _collect_runner_labels(value, labels)
    elif isinstance(node, list):
        for item in node:
            _collect_runner_labels(item, labels)


def _values_are_safe(node):
    """True when every string leaf matches the shell-safe charset."""
    if isinstance(node, str):
        return bool(node) and not (set(node) - _SAFE_VALUE_CHARS)
    if isinstance(node, dict):
        return all(_values_are_safe(value) for value in node.values())
    if isinstance(node, list):
        return all(_values_are_safe(item) for item in node)
    return True


def _validate_entries(entries, allowed_runners):
    """Filter PR-derived matrix entries against the base-branch allowlist.

    Drops entries whose runner label does not exist in the base config or
    whose values carry shell metacharacters, so a crafted PR config cannot
    retarget jobs to arbitrary runners or smuggle commands into downstream
    run: blocks. Legit PRs only add or pick test groups, which still pass.
    """
    validated = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        labels = set()
        _collect_runner_labels(entry, labels)
        if labels and not labels <= allowed_runners:
            print(f"::notice::Dropped matrix entry with unknown runner label(s): {sorted(labels)}")
            continue
        if not _values_are_safe(entry):
            print("::notice::Dropped matrix entry with unsafe characters in values")
            continue
        validated.append(entry)
    return validated


def cmd_dispatch(_args):
    """Resolve /nightly <name> tokens and emit dispatch flags.

    Reads NIGHTLY_MATRIX (base64-encoded nightly_config.yaml) and TEST_CASES
    (comma-separated tokens from the /nightly comment).     Writes to GITHUB_OUTPUT:
      - dispatch_a2=true|false
      - dispatch_a3=true|false
      - dispatch_a3_560t=true|false
      - dispatch_310p=true|false
      - dispatch_a5=true|false
      - test_cases=<group,model>  (only when a /<model> token matched an accuracy group)
    """
    # Prefer NIGHTLY_MATRIX for backward compatibility; fall back to WEEKLY_MATRIX
    # so the same script can resolve test names from either config family.
    matrix_b64 = os.environ.get("NIGHTLY_MATRIX", "") or os.environ.get("WEEKLY_MATRIX", "")
    a2_names = parse_nightly_matrix(matrix_b64, "a2")
    a3_names = parse_nightly_matrix(matrix_b64, "a3")
    a3_560t_names = parse_nightly_matrix(matrix_b64, "a3-560t")
    _310p_names = parse_nightly_matrix(matrix_b64, "310p")
    a5_names = parse_nightly_matrix(matrix_b64, "a5")

    is_a3_560t = os.environ.get("IS_A3_560T", "") == "true"

    raw = os.environ.get("TEST_CASES", "")
    test_cases = [tc.strip() for tc in raw.split(",") if tc.strip()]

    da2, da3, da3_560t, d310p, da5 = False, False, False, False, False
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
            if is_a3_560t:
                if tc in a3_560t_names:
                    da3_560t = True
            else:
                if tc in a2_names:
                    da2 = True
                if tc in a3_names:
                    da3 = True
                if tc in _310p_names:
                    d310p = True
                if tc in a5_names:
                    da5 = True

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        if transformed:
            f.write(f"test_cases={transformed}\n")
        f.write(f"dispatch_a2={str(da2).lower()}\n")
        f.write(f"dispatch_a3={str(da3).lower()}\n")
        f.write(f"dispatch_a3_560t={str(da3_560t).lower()}\n")
        f.write(f"dispatch_310p={str(d310p).lower()}\n")
        f.write(f"dispatch_a5={str(da5).lower()}\n")


# Workflows that support PR-derived configs check the PR out under ./pr/.
_PR_CHECKOUT_PREFIX = "./pr/"


def _resolve_allowlist_file(matrix_file):
    """Base-branch config that a PR-derived matrix must be checked against.

    The default maps a ./pr/ checkout path to the same relative path in the
    base-branch checkout, so every caller reading a PR-derived MATRIX_FILE is
    covered without wiring an allowlist per workflow; MATRIX_ALLOWLIST_FILE
    overrides it. Returns "" when no allowlist applies.
    """
    explicit = os.environ.get("MATRIX_ALLOWLIST_FILE", "")
    if explicit:
        return explicit
    if matrix_file.startswith(_PR_CHECKOUT_PREFIX):
        return matrix_file[len(_PR_CHECKOUT_PREFIX) :]
    return ""


def cmd_matrix(_args):
    """Extract matrix sections per MATRIX_OUTPUTS spec.

    Reads MATRIX_FILE (path to nightly_config.yaml) and MATRIX_OUTPUTS
    (JSON object {output_name: "dot.path.in.yaml"}). Writes to GITHUB_OUTPUT:
      - <output_name>=<json array>  (one entry per spec item)

    A MATRIX_FILE pointing at the PR checkout (./pr/...) is validated against
    the base-branch config at the same relative path: unknown runner labels
    are dropped and shell metacharacters are rejected, so a crafted PR config
    cannot retarget jobs to arbitrary runners or smuggle commands into
    downstream run blocks. The trusted base config itself is never filtered,
    so normal nightly runs cannot silently lose coverage.
    """
    matrix_file = os.environ.get("MATRIX_FILE", "")
    if not matrix_file:
        raise SystemExit("MATRIX_FILE env var is required for --mode=matrix")
    spec = json.loads(os.environ.get("MATRIX_OUTPUTS", "{}"))

    with open(matrix_file) as f:
        matrix = yaml.safe_load(f) or {}

    allowlist_file = _resolve_allowlist_file(matrix_file)
    validate = False
    allowed_runners = set()
    if allowlist_file and os.path.realpath(allowlist_file) != os.path.realpath(matrix_file):
        validate = True
        with open(allowlist_file) as f:
            _collect_runner_labels(yaml.safe_load(f) or {}, allowed_runners)

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for name, path in spec.items():
            node = _walk(matrix, path)
            cfg = node if isinstance(node, list) else []
            if validate:
                cfg = _validate_entries(cfg, allowed_runners)
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
