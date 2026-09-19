#!/usr/bin/env bash
set -euo pipefail

image="${1:?image reference required}"
cache_dir="${2:?host cache directory required}"
container_cache_path="${3:?container cache path required}"

tmp_dir="${cache_dir}.export-$$"
cid=""

cleanup() {
  set +e
  if [ -n "$cid" ]; then
    docker rm -f "$cid" >/dev/null 2>&1 || true
  fi
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

cid="$(docker create "$image")"
docker cp "${cid}:${container_cache_path}/." "$tmp_dir/"

rm -rf "$cache_dir"
mv "$tmp_dir" "$cache_dir"
tmp_dir=""
du -sh "$cache_dir" || true
