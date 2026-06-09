import base64
import json
import os
import subprocess

try:
    import yaml
except ImportError:
    subprocess.check_call(["pip", "install", "pyyaml", "-q"])
    import yaml


def parse_names(b64_content):
    if not b64_content:
        return set()
    try:
        content = base64.b64decode(b64_content).decode()
        parsed = yaml.safe_load(content)
        names = set()
        for job in parsed.get("jobs", {}).values():
            tc = job.get("strategy", {}).get("matrix", {}).get("test_config", [])
            if isinstance(tc, list):
                for entry in tc:
                    if isinstance(entry, dict) and "name" in entry:
                        names.add(entry["name"])
        return names
    except Exception:
        return set()


def parse_accuracy_names(b64_content):
    if not b64_content:
        return set()
    try:
        parsed = json.loads(base64.b64decode(b64_content).decode())
        names = set()
        for group_list in parsed.values():
            if isinstance(group_list, list):
                for group in group_list:
                    if isinstance(group, dict) and "name" in group:
                        names.add(group["name"])
        return names
    except Exception:
        return set()


def main():
    a2_names = parse_names(os.environ.get("A2_RAW", ""))
    a2_names |= parse_accuracy_names(os.environ.get("A2_ACC_GROUPS", ""))
    a3_names = parse_names(os.environ.get("A3_RAW", ""))

    raw_test_cases = os.environ.get("TEST_CASES", "")
    test_cases = [tc.strip() for tc in raw_test_cases.split(",") if tc.strip()]

    da2, da3 = False, False
    transformed_tc = None
    for tc in test_cases:
        if "/" in tc:
            parts = tc.split("/", 1)
            group_name, model_name = parts[0], parts[1]
            if group_name in a2_names:
                da2 = True
                transformed_tc = f"{group_name},{model_name}"
            break
        elif tc == "accuracy-group":
            da2 = True
        else:
            if tc in a2_names:
                da2 = True
            if tc in a3_names:
                da3 = True

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        if transformed_tc:
            f.write(f"test_cases={transformed_tc}\n")
        f.write(f"dispatch_a2={str(da2).lower()}\n")
        f.write(f"dispatch_a3={str(da3).lower()}\n")


if __name__ == "__main__":
    main()
