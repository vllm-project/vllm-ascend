import os
import hashlib

dir1 = r"D:\vllm-ascend-rfc-vllm_cann"
dir2 = r"C:\Users\qigs\Downloads\vllm-ascend-rfc-vllm_cann (3)\vllm-ascend-rfc-vllm_cann"

exts = {'.py', '.cpp', '.h', '.hpp', '.json', '.yaml', '.yml', '.md', '.txt', '.sh', '.cmake', '.in'}

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def scan_dir(root_dir):
    file_hashes = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '.git' in dirnames:
            dirnames.remove('.git')
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in exts:
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                try:
                    file_hashes[rel_path] = get_file_hash(full_path)
                except Exception:
                    pass
    return file_hashes

print("Scanning D drive...")
hashes1 = scan_dir(dir1)
print(f"Found {len(hashes1)} files in D drive.")

print("Scanning Downloads drive...")
hashes2 = scan_dir(dir2)
print(f"Found {len(hashes2)} files in Downloads drive.")

new_files = []
mod_files = []

for rel_path, hash1 in hashes1.items():
    if rel_path not in hashes2:
        new_files.append(rel_path)
    elif hash1 != hashes2[rel_path]:
        mod_files.append(rel_path)

print(f"\n=== NEW FILES ({len(new_files)}) ===")
for f in sorted(new_files):
    print(f"[NEW] {f}")

print(f"\n=== MODIFIED FILES ({len(mod_files)}) ===")
for f in sorted(mod_files):
    print(f"[MOD] {f}")

print(f"\nTotal: {len(new_files)} new, {len(mod_files)} modified.")
