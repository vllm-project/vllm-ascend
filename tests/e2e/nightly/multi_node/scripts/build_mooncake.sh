#!/bin/bash

set -e
set -o pipefail

GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

branch=${1:-v0.3.7.post2}

repo_url="https://github.com/kvcache-ai/Mooncake"
repo_name="Mooncake"
state_file=".build_state"

echo "[INFO] Branch: $branch"
echo "-------------------------------------------"


mark_done() { echo "$1" >> "$state_file"; }
is_done() { grep -Fxq "$1" "$state_file" 2>/dev/null; }

if ! is_done "clone"; then
  echo "[STEP] Clone repository..."
  if [ -d "$repo_name" ]; then
    echo "[WARN] Directory $repo_name already exists, skipping clone."
  else
    git clone --branch "$branch" --depth 1 "$repo_url" "$repo_name"
  fi
  mark_done "clone"
else
  echo "[SKIP] Clone step already done."
fi

init_ascend_env() {
    cann_in_sys_path=/usr/local/Ascend/ascend-toolkit; \
    cann_in_user_path=$HOME/Ascend/ascend-toolkit; \
    uname_m=$(uname -m) && \
    if [ -f "${cann_in_sys_path}/set_env.sh" ]; then \
        source ${cann_in_sys_path}/set_env.sh; \
        export LD_LIBRARY_PATH=${cann_in_sys_path}/latest/lib64:${cann_in_sys_path}/latest/${uname_m}-linux/devlib:${LD_LIBRARY_PATH} ; \
    elif [ -f "${cann_in_user_path}/set_env.sh" ]; then \
        source "$HOME/Ascend/ascend-toolkit/set_env.sh"; \
        export LD_LIBRARY_PATH=${cann_in_user_path}/latest/lib64:${cann_in_user_path}/latest/${uname_m}-linux/devlib:${LD_LIBRARY_PATH}; \ 
    else \
        echo "No Ascend Toolkit found"; \
        exit 1; \
    fi
}

init_ascend_env

if ! is_done "deps"; then
  cd "$repo_name"
  echo "[STEP]Installing dependencies..."
  sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
  bash dependencies.sh -y
  cd ..
  mark_done "deps"
else
  echo "[SKIP] Dependencies already installed."
fi


if ! is_done "mpi"; then
  echo "[STEP] Install MPI..."
  apt purge -y mpich libmpich-dev openmpi-bin libopenmpi-dev || true
  apt install -y mpich libmpich-dev
  export CPATH=/usr/lib/aarch64-linux-gnu/mpich/include/:${CPATH:-}
  export CPATH=/usr/lib/aarch64-linux-gnu/openmpi/lib:${CPATH:-}
  mark_done "mpi"
else
  echo "[SKIP] MPI installation already done."
fi


if ! is_done "build"; then
  echo "[STEP] Compile and install..."
  cd "$repo_name"

  if [ -d "build" ]; then
    echo "[INFO] Removing existing build directory..."
    rm -rf build
  fi

  mkdir build && cd build
  cmake .. -USE_ASCEND_DIRECT=ON || { echo "[ERROR] cmake failed."; exit 1; }
  make -j || { echo "[ERROR] make failed."; exit 1; }
  make install || { echo "[ERROR] make install failed."; exit 1; }
  mark_done "build"
else
  echo "[SKIP] Build already done."
fi


if ! grep -q "export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH" ~/.bashrc; then
    echo -e "${YELLOW}Adding LD_LIBRARY_PATH to your PATH in ~/.bashrc${NC}"
    echo 'export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo -e "${YELLOW}Please run 'source ~/.bashrc' or start a new terminal${NC}"
fi
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH


echo "=========================================="
echo -e "${GREEN}[SUCCESS] Mooncake build completed!"
echo "You can rerun this script anytime — it will resume from the last step."
echo "=========================================="

echo "Example startup command:"
echo "mooncake_master --eviction_high_watermark_ratio 0.8 --eviction_ratio 0.05 --port 50088"
