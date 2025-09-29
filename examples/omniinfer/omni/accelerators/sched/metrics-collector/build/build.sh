#!/bin/bash

# Go 安装脚本
set -e

GO_VERSION="1.22.1"
INSTALL_DIR="/usr/local"
TAR_FILE="go${GO_VERSION}.linux-amd64.tar.gz"
DOWNLOAD_URL="https://golang.google.cn/dl/${TAR_FILE}"

# 检查是否已安装指定版本
if [ -d "${INSTALL_DIR}/go" ]; then
    INSTALLED_VERSION=$("${INSTALL_DIR}/go/bin/go" version | awk '{print $3}' | sed 's/go//')
    if [ "$INSTALLED_VERSION" = "$GO_VERSION" ]; then
        echo "Go ${GO_VERSION} 已经安装"
        exit 0
    else
        echo "发现已安装的 Go 版本: ${INSTALLED_VERSION}"
        echo "将升级到版本: ${GO_VERSION}"
    fi
fi

# 创建临时目录
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

echo "正在下载 Go ${GO_VERSION}..."
if command -v wget &> /dev/null; then
    wget "$DOWNLOAD_URL"
elif command -v curl &> /dev/null; then
    curl -LO "$DOWNLOAD_URL"
else
    echo "错误: 需要 wget 或 curl 来下载文件"
    exit 1
fi

# 检查下载是否成功
if [ ! -f "$TAR_FILE" ]; then
    echo "错误: 下载失败"
    exit 1
fi

echo "正在安装到 ${INSTALL_DIR}..."
sudo tar -C "$INSTALL_DIR" -xzf "$TAR_FILE"

# 清理临时文件
cd -
rm -rf "$TMP_DIR"

# 设置环境变量提示
echo ""
echo "Go ${GO_VERSION} 安装完成！"
echo ""
echo "请将以下内容添加到 ~/.bashrc 或 ~/.zshrc 文件中："
echo ""
echo "export GOPATH=\$HOME/go"
echo "export PATH=\$PATH:${INSTALL_DIR}/go/bin:\$GOPATH/bin"
echo ""
echo "然后运行: source ~/.bashrc 或 source ~/.zshrc"


# 设置环境变量（当前会话有效）
export GOROOT="${INSTALL_DIR}/go"
export GOPATH="$HOME/go"
export PATH="$GOROOT/bin:$GOPATH/bin:$PATH"


# 验证安装
if [ -f "${INSTALL_DIR}/go/bin/go" ]; then
    echo ""
    echo "验证安装:"
    "${INSTALL_DIR}/go/bin/go" version
else
    echo "警告: 安装可能未成功完成"
fi

# 安装metrics-collector
# 进入指定目录并执行 go build 和 go install
set -e

TARGET_DIR="../cmd/metrics-collector"

echo "正在进入目录: $TARGET_DIR"
cd "$TARGET_DIR" || { echo "错误: 无法进入目录 $TARGET_DIR"; exit 1; }

echo "当前工作目录: $(pwd)"
echo "执行 go build..."
go build

echo "执行 go install..."
go install

echo "操作完成！"

