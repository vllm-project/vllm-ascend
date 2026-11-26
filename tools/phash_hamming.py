"""Measure pairwise pHash Hamming distances across images (pure Python, no
torch dependency).

The algorithm mirrors vllm_ascend/distributed/ec_transfer/phash.py:

  1. Resize the image to (gh*14, gw*14) and average each 14x14 patch to one
     gray value (the patch mean is order-independent, so the merge
     permutation is not undone here);
  2. Bilinear resize to 32x32 (align_corners=False, matching torch);
  3. DCT-II, keeping the top-left 8x8 low frequencies;
  4. Compare against the median; bits within |coef - median| <= 0.05*std
     are forced to 0.

This script uses float64 while phash.py uses float32, so a few bits near the
median may differ; the dead zone suppresses most of that drift.

Dependency: Pillow. Falls back to PowerShell + System.Drawing when Pillow is
unavailable.

Usage:

    python tools/phash_hamming.py --images-dir /path/to/images
    python tools/phash_hamming.py --images-dir /path/to/images --gh 16 --gw 16
"""

import argparse
import math
import os
import sys

# When run from tools/, the script directory lands in sys.path[0] and
# tools/bisect would shadow the stdlib bisect module (an indirect dependency
# of tempfile/random). Pop it (this script imports no sibling modules). Must
# run before "import tempfile".
_script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _script_dir:
    sys.path.pop(0)

import subprocess  # noqa: E402
import tempfile  # noqa: E402

PATCH_SIZE = 14  # Qwen2/3-VL patch edge size
MERGE_SIZE = 2  # Qwen2/3-VL spatial merge size (only for validating gh/gw)
# The following three constants mirror phash.py
PHASH_SIZE = 32  # DCT input size
HASH_SIDE = 8  # Keep the top-left 8x8 low frequencies -> 64 bits
DEAD_ZONE = 0.05  # Median dead-zone ratio

# DCT-II matrix; row 0 is scaled by 1/sqrt(2) as in phash.py
_DCT = [[math.cos(math.pi / PHASH_SIZE * (i + 0.5) * k) for i in range(PHASH_SIZE)] for k in range(PHASH_SIZE)]
for _k in range(PHASH_SIZE):
    _DCT[0][_k] *= 1.0 / math.sqrt(2.0)


def hamming(a: int, b: int) -> int:
    """Hamming distance between two pHashes (same as phash.py)."""
    return bin(a ^ b).count("1")


def bilinear_resize(gray: list[list[float]], out: int) -> list[list[float]]:
    """Bilinear resize to out x out (align_corners=False, matching torch
    F.interpolate)."""
    h, w = len(gray), len(gray[0])
    scale_h, scale_w = h / out, w / out
    res = []
    for oy in range(out):
        sy = (oy + 0.5) * scale_h - 0.5
        y0 = min(max(int(math.floor(sy)), 0), h - 1)
        y1 = min(y0 + 1, h - 1)
        fy = min(max(sy - math.floor(sy), 0.0), 1.0)
        row = []
        for ox in range(out):
            sx = (ox + 0.5) * scale_w - 0.5
            x0 = min(max(int(math.floor(sx)), 0), w - 1)
            x1 = min(x0 + 1, w - 1)
            fx = min(max(sx - math.floor(sx), 0.0), 1.0)
            row.append(
                gray[y0][x0] * (1 - fx) * (1 - fy)
                + gray[y0][x1] * fx * (1 - fy)
                + gray[y1][x0] * (1 - fx) * fy
                + gray[y1][x1] * fx * fy
            )
        res.append(row)
    return res


def compute_phash(gray: list[list[float]]) -> int:
    """Compute a 64-bit pHash from a (gh, gw) patch gray plane (mirrors
    phash.py)."""
    img = bilinear_resize(gray, PHASH_SIZE)
    n = PHASH_SIZE
    # dct = _DCT @ img @ _DCT.T
    tmp = [[sum(_DCT[i][k] * img[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    dct = [[sum(tmp[i][k] * _DCT[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    low = [dct[i][j] for i in range(HASH_SIDE) for j in range(HASH_SIDE)]

    srt = sorted(low)
    # torch Tensor.median() takes the lower median for even lengths
    median = srt[len(low) // 2 - 1]
    mean = sum(low) / len(low)
    std = math.sqrt(sum((v - mean) ** 2 for v in low) / len(low))
    dead = DEAD_ZONE * std
    phash = 0
    for v in low:
        bit = 1 if (v > median and abs(v - median) > dead) else 0
        phash = (phash << 1) | bit
    return phash


# None = not probed yet; True/False = PIL availability
_use_pil: bool | None = None


def _ps_quote(path: str) -> str:
    """Escape a PowerShell single-quoted string."""
    return "'" + path.replace("'", "''") + "'"


def _load_resized_via_powershell(img_path: str, out_w: int, out_h: int) -> list[list[tuple[int, int, int]]]:
    """Fallback when PIL is unavailable: resize via PowerShell System.Drawing,
    save as BMP, and parse it in pure Python."""
    tmp_dir = tempfile.mkdtemp(prefix="phash_")
    bmp_path = os.path.join(tmp_dir, "out.bmp")
    ps_cmd = (
        "Add-Type -AssemblyName System.Drawing; "
        f"$src = [System.Drawing.Bitmap]::FromFile({_ps_quote(os.path.abspath(img_path))}); "
        f"$dst = New-Object System.Drawing.Bitmap({out_w}, {out_h}); "
        "$g = [System.Drawing.Graphics]::FromImage($dst); "
        "$g.InterpolationMode = "
        "[System.Drawing.Drawing2D.InterpolationMode]::Bilinear; "
        f"$g.DrawImage($src, 0, 0, {out_w}, {out_h}); "
        f"$dst.Save({_ps_quote(bmp_path)}, [System.Drawing.Imaging.ImageFormat]::Bmp); "
        "$g.Dispose(); $dst.Dispose(); $src.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd], check=True, capture_output=True)

    with open(bmp_path, "rb") as f:
        data = f.read()
    if data[:2] != b"BM":
        raise RuntimeError(f"PowerShell output is not a BMP: {img_path}")
    off = int.from_bytes(data[10:14], "little")
    w = int.from_bytes(data[18:22], "little", signed=True)
    h = int.from_bytes(data[22:26], "little", signed=True)
    bpp = int.from_bytes(data[28:30], "little")
    comp = int.from_bytes(data[30:34], "little")
    if bpp not in (24, 32) or comp != 0:
        raise RuntimeError(f"only 24/32-bit uncompressed BMP is supported, got bpp={bpp} comp={comp}")
    px_size = bpp // 8
    stride = (w * px_size + 3) // 4 * 4
    rows = []
    for r in range(abs(h)):
        # BMP rows are stored bottom-up when height > 0
        base = off + (abs(h) - 1 - r if h > 0 else r) * stride
        row = []
        for x in range(w):
            b, g, rr = data[base + x * px_size : base + x * px_size + 3]
            row.append((rr, g, b))
        rows.append(row)
    os.remove(bmp_path)
    os.rmdir(tmp_dir)
    return rows


def _load_resized_via_pil(img_path: str, out_w: int, out_h: int) -> list[list[tuple[int, int, int]]]:
    from PIL import Image

    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((out_w, out_h), Image.BILINEAR)
        px = im.load()
        return [[px[x, y] for x in range(out_w)] for y in range(out_h)]


def image_to_gray(img_path: str, gh: int, gw: int) -> list[list[float]]:
    """Image -> (gh, gw) patch gray plane: bilinearly resize to
    (gh*14, gw*14) (mirroring the Qwen2/3-VL resize), then average each
    14x14x3 patch."""
    out_w, out_h = gw * PATCH_SIZE, gh * PATCH_SIZE
    global _use_pil
    if _use_pil is None:
        try:
            import PIL.Image  # noqa: F401

            _use_pil = True
        except (ImportError, OSError):
            _use_pil = False
            print("(PIL unavailable, falling back to PowerShell System.Drawing decoding)")
    rows = (
        _load_resized_via_pil(img_path, out_w, out_h)
        if _use_pil
        else _load_resized_via_powershell(img_path, out_w, out_h)
    )
    gray = []
    for r in range(gh):
        row = []
        for c in range(gw):
            acc = 0
            for y in range(r * PATCH_SIZE, (r + 1) * PATCH_SIZE):
                for pix in rows[y][c * PATCH_SIZE : (c + 1) * PATCH_SIZE]:
                    acc += pix[0] + pix[1] + pix[2]
            row.append(acc / (PATCH_SIZE * PATCH_SIZE * 3))
        gray.append(row)
    return gray


def load_gray_planes(images_dir: str, gh: int, gw: int) -> dict[str, list[list[float]]]:
    planes = {}
    for name in sorted(os.listdir(images_dir)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        planes[name] = image_to_gray(os.path.join(images_dir, name), gh, gw)
    return planes


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure pairwise pHash Hamming distances across images")
    parser.add_argument("--images-dir", required=True, help="image directory (jpg/jpeg/png/bmp/webp)")
    parser.add_argument(
        "--gh", type=int, default=16, help=f"patch grid height, must be a multiple of {MERGE_SIZE} (default 16)"
    )
    parser.add_argument(
        "--gw", type=int, default=16, help=f"patch grid width, must be a multiple of {MERGE_SIZE} (default 16)"
    )
    args = parser.parse_args()

    if args.gh % MERGE_SIZE or args.gw % MERGE_SIZE:
        sys.exit(f"--gh/--gw must be divisible by merge_size={MERGE_SIZE}")

    planes = load_gray_planes(args.images_dir, args.gh, args.gw)
    if not planes:
        sys.exit(f"no images found in directory: {args.images_dir}")

    print(f"images dir : {args.images_dir}")
    print(f"grid (t,h,w): (1, {args.gh}, {args.gw}), merge_size={MERGE_SIZE}\n")

    phashes: dict[str, int] = {}
    for name, gray in planes.items():
        phashes[name] = compute_phash(gray)
        print(f"  {name:<40} 0x{phashes[name]:016x}")

    names = list(phashes)
    print("\npairwise hamming distance matrix:")
    col_w = max(len(n) for n in names) + 2
    short = [n[:12] for n in names]
    print(" " * (col_w + 1) + " ".join(f"{s:>12}" for s in short))
    for a, sa in zip(names, short):
        row = [f"{hamming(phashes[a], phashes[b]):>12}" for b in names]
        print(f"{sa:>{col_w}} " + " ".join(row))

    base = names[0]
    print(f"\ndistances from base {base} (ascending):")
    for n in sorted(names, key=lambda n: hamming(phashes[base], phashes[n])):
        print(f"  {hamming(phashes[base], phashes[n]):>3}  {n}")


if __name__ == "__main__":
    main()
