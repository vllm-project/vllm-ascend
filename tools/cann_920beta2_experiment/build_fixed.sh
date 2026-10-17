#!/usr/bin/env bash
set -euo pipefail

# Only for the disposable A3 job, with the matched CANN environment already active.
evidence=$(realpath "$1")
[[ $(uname -m) == aarch64 ]]
for dependency in git cmake g++ pigz dos2unix; do
    command -v "$dependency" || { echo "Missing build tool after preparation: $dependency" >&2; exit 1; }
done
src=$(cat "$evidence/fix-source-dir.txt")
[[ $(git -C "$src" rev-parse HEAD) == 30ef7dd563c8a4b74c3161835c8e47d1d96f87b6 ]]
# Recheck the complete staged tree and reject unstaged changes before compiling.
[[ $(git -C "$src" write-tree) == c41d331cdab95233834dbca87e0ada238251a5f0 ]]
git -C "$src" diff --quiet
cd "$src"
bash build.sh --pkg --soc=ascend910_93 --ops=add_rms_norm_dynamic_quant \
    --vendor_name=ardqv2_fix_920 --ccache=off -j8 \
    > "$evidence/fix-build.log" 2>&1

mkdir -p "$evidence/fix-packages"
mapfile -t packages < <(find "$src/build_out" -maxdepth 1 -type f -name '*.run')
[[ ${#packages[@]} == 1 ]]
cp "${packages[0]}" "$evidence/fix-packages/"
sha256sum "$evidence"/fix-packages/*.run > "$evidence/fix-package-sha256.txt"
install_dir=$(mktemp -d /opt/cann920-fixed-vendor.XXXXXX)
chmod 755 "$install_dir"
bash "${packages[0]}" --quiet --install-path="$install_dir" > "$evidence/fix-install.log" 2>&1
mapfile -t environments < <(find "$install_dir/vendors" -type f -path '*/bin/set_env.bash')
[[ ${#environments[@]} == 1 ]]
vendor=$(dirname "$(dirname "${environments[0]}")")
printf 'source %q\n' "${environments[0]}" > "$evidence/fix-runtime-env.sh"
printf 'export ARDQ_EXPECTED_VENDOR=%q\n' "$vendor" >> "$evidence/fix-runtime-env.sh"
printf 'export ARDQ_EXPECTED_LIBRARIES=%q\n' "$evidence/fix-install-libraries.json" >> "$evidence/fix-runtime-env.sh"
python3 - "$src" "$vendor" "$evidence" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

src, vendor, evidence = map(Path, sys.argv[1:])
hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in vendor.rglob("*.so") if p.is_file()}
(evidence / "fix-install-libraries.json").write_text(json.dumps(hashes, indent=2) + "\n")
metadata = {
    "base_commit": "30ef7dd563c8a4b74c3161835c8e47d1d96f87b6",
    "validated_fix_commit": "9f85253facda2bab7b38924b3665f1d8089b7f15",
    "patched_tree": subprocess.check_output(["git", "-C", str(src), "write-tree"], text=True).strip(),
    "soc": "ascend910_93", "vendor": str(vendor), "build_returncode": 0, "install_returncode": 0,
    "dependencies": {},
}
for name, relative in {"opbase": "third_party/opbase", "cann-cmake": "build/_deps/cann-cmake-src"}.items():
    metadata["dependencies"][name] = subprocess.check_output(
        ["git", "-C", str(src / relative), "rev-parse", "HEAD"], text=True).strip()
(evidence / "fix-build-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata), flush=True)
PY
