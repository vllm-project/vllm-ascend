#!/bin/sh
# test-case-006: squid 代理下载测试（GitHub / Gitee）
#
# 对比「走 squid 代理」vs「不走代理」两种情况下，从 GitHub / Gitee 下载文件是否稳定。
# POSIX sh 兼容（不使用 bashism），因为实际的 job 容器镜像不保证是 bash。
#
# 用法：
#   sh test-download.sh
#
# 环境变量（可选覆盖默认值）：
#   SQUID_PROXY   默认 http://squid-cache.squid.svc.cluster.local:3128
#   SQUID_CA      默认 /etc/squid-ca/squid-ca.pem
#   MAX_TIME      单次 curl 超时秒数，默认 30
#   MAX_REDIRS    单次 curl 最大重定向次数，默认 10（保留有限值方便观察重定向循环，而不是等到默认 50 次）

SQUID_PROXY="${SQUID_PROXY:-http://squid-cache.squid.svc.cluster.local:3128}"
SQUID_CA="${SQUID_CA:-/etc/squid-ca/squid-ca.pem}"
MAX_TIME="${MAX_TIME:-30}"
MAX_REDIRS="${MAX_REDIRS:-10}"

GITHUB_URL="https://github.com/moby/buildkit/releases/download/v0.29.0/buildkit-v0.29.0.linux-arm64.tar.gz"
GITEE_URL="https://gitee.com/mirrors/buildkit/releases/download/v0.29.0/buildkit-v0.29.0.linux-arm64.tar.gz"
GITEE_FALLBACK_URL="https://gitee.com"

RESULT_SUMMARY=""

# ── best-effort：安装 squid CA，装不上就继续（不阻塞测试本身）──
install_squid_ca() {
  if [ -f "$SQUID_CA" ]; then
    if command -v update-ca-certificates >/dev/null 2>&1; then
      if cp "$SQUID_CA" /usr/local/share/ca-certificates/squid-ca.crt 2>/dev/null; then
        if update-ca-certificates >/dev/null 2>&1; then
          echo "[setup] squid CA installed."
        else
          echo "[setup] WARNING: update-ca-certificates failed." >&2
        fi
      else
        echo "[setup] WARNING: failed to copy squid CA (permission denied?)." >&2
      fi
    else
      echo "[setup] WARNING: update-ca-certificates not found; skipping CA install." >&2
    fi
  else
    echo "[setup] WARNING: $SQUID_CA not found; skipping CA install." >&2
  fi
}

# 单次下载测试。
# args: label url use_proxy(yes/no) outfile
run_case() {
  label="$1"
  url="$2"
  use_proxy="$3"
  outfile="$4"

  echo ""
  echo "==================================================================="
  echo "[$label] url=$url use_proxy=$use_proxy"
  echo "==================================================================="

  if [ "$use_proxy" = "yes" ]; then
    export HTTP_PROXY="$SQUID_PROXY"
    export HTTPS_PROXY="$SQUID_PROXY"
    export http_proxy="$SQUID_PROXY"
    export https_proxy="$SQUID_PROXY"
  else
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
  fi

  rm -f "$outfile"
  start_ts=$(date +%s)
  # -v 输出到临时文件，事后只挑关键行打印，避免刷屏；同时保留完整 log 供需要时查阅。
  verbose_log="/tmp/curl-verbose-$$.log"
  curl -fsSL -v --max-time "$MAX_TIME" --max-redirs "$MAX_REDIRS" -o "$outfile" "$url" >"$verbose_log" 2>&1
  exit_code=$?
  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))

  echo "--- key verbose lines ---"
  grep -iE "^< location|^> GET|^> CONNECT|HTTP/[12](\.[012])? [0-9]{3}|ssl certificate|too many redirects|operation timed out|connection refused|proxy replied" "$verbose_log" 2>/dev/null || true

  if [ "$exit_code" -eq 0 ] && [ -s "$outfile" ]; then
    size=$(wc -c <"$outfile" 2>/dev/null | tr -d ' ')
    echo "--- RESULT: OK  exit=$exit_code  elapsed=${elapsed}s  size=${size} bytes ---"
    RESULT_SUMMARY="${RESULT_SUMMARY}${label}: OK (exit=$exit_code, ${elapsed}s, ${size}B)\n"
  else
    echo "--- RESULT: FAIL  exit=$exit_code  elapsed=${elapsed}s ---"
    RESULT_SUMMARY="${RESULT_SUMMARY}${label}: FAIL (exit=$exit_code, ${elapsed}s)\n"
  fi

  rm -f "$verbose_log" "$outfile"
}

echo "########################################################"
echo "# test-case-006: GitHub / Gitee download via squid proxy"
echo "########################################################"
echo "SQUID_PROXY=$SQUID_PROXY"
echo "MAX_TIME=${MAX_TIME}s  MAX_REDIRS=$MAX_REDIRS"
echo ""

install_squid_ca

run_case "github-no-proxy"   "$GITHUB_URL" "no"  /tmp/github-no-proxy.tar.gz
run_case "github-via-proxy" "$GITHUB_URL" "yes" /tmp/github-via-proxy.tar.gz

run_case "gitee-no-proxy"   "$GITEE_URL" "no"  /tmp/gitee-no-proxy.tar.gz
run_case "gitee-via-proxy" "$GITEE_URL" "yes" /tmp/gitee-via-proxy.tar.gz

# gitee release 资源路径不一定存在（mirrors/buildkit 是否有该 tag 未知），
# 额外测一次首页连通性作为兜底，避免因为 URL 本身 404 而误判为代理问题。
run_case "gitee-homepage-no-proxy"   "$GITEE_FALLBACK_URL" "no"  /tmp/gitee-home-no-proxy.html
run_case "gitee-homepage-via-proxy" "$GITEE_FALLBACK_URL" "yes" /tmp/gitee-home-via-proxy.html

echo ""
echo "########################################################"
echo "# SUMMARY"
echo "########################################################"
printf "%b" "$RESULT_SUMMARY"
