#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python - <<'PY'
import hashlib
import importlib.metadata
import json
import pathlib
expected = json.loads(pathlib.Path('native_manifest.json').read_text())
if importlib.metadata.version('memcache-hybrid') != expected['version']:
    raise RuntimeError('Review changed Memcache version')
if hashlib.sha256(pathlib.Path(expected['library']).read_bytes()).hexdigest() != expected['sha256']:
    raise RuntimeError('Review changed Memcache library')
PY
memcache_lib=$(python -c 'import json; print(json.load(open("native_manifest.json"))["library"].rsplit("/", 1)[0])')
python_lib=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
python_name=$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
g++ -shared -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 native_memcache_key_parser.cpp -o libmemcache_event_key_fix.so
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 test_native_parser.cpp -L"$memcache_lib" -lmf_memcache -L"$python_lib" -l"$python_name" -Wl,-rpath,"$memcache_lib" -o test_native_parser
LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$PWD/libmemcache_event_key_fix.so" ./test_native_parser
g++ -shared -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 test_native_local_loader.cpp -L"$memcache_lib" -lmf_memcache -Wl,-rpath,"$memcache_lib" -o test_native_local_loader.so
LD_PRELOAD="$PWD/libmemcache_event_key_fix.so" python test_native_local_loader.py
