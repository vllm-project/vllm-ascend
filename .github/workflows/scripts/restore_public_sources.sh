#!/bin/bash
# Restore public package sources so externally-published images can install
# packages; the internal mirrors (cache-service) are only needed during build.
set -e

# pip: reset index-url to a public mirror and drop the internal trusted host.
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip config unset global.trusted-host 2>/dev/null || true

# cargo: drop the internal crates.io mirror / git-proxy config written during build.
rm -f /root/.cargo/config.toml

# git: drop any internal GitHub proxy insteadOf rule.
for key in $(git config --global --name-only --get-regexp '^url\..*\.insteadof$' 2>/dev/null); do
    git config --global --unset-all "$key"
done
