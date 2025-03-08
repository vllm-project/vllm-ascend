# 贡献指南

## 构建与测试
我们推荐您在提交PR之前在本地开发环境进行构建和测试。

### 环境准备与构建
理论上，vllm-ascend 构建仅支持 Linux，因为`vllm-ascend` 依赖项 `torch_npu` 仅支持 Linux。

但是您仍然可以在 Linux/Windows/macOS 上配置开发环境进行代码检查和基本测试，如下命令所示：

```bash
# 选择基础文件夹 (~/vllm-project/) ，创建python虚拟环境
cd ~/vllm-project/
python3 -m venv .venv
source ./.venv/bin/activate

# 克隆并安装vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE="empty" pip install .
cd ..

# 克隆并安装vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -r requirements-dev.txt

# 通过执行以下脚本以运行 lint 及 mypy 测试
bash format.sh

# 构建:
# - 目前仅支持在Linux上进行完整构建（torch_npu 限制）
# pip install -e .
# - 在其他操作系统上构建安装，需要跳过依赖
# - build without deps for debugging in other OS
# pip install -e . --no-deps

# 使用 `-s` 提交更改
git commit -sm "your commit info"
```

### 测试
虽然 vllm-ascend CI 提供了对 [Ascend](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/vllm_ascend_test.yaml) 的集成测试，但您也可以在本地运行它。在本地运行这些集成测试的最简单方法是通过容器：

```bash
# 基于昇腾NPU环境
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

export IMAGE=vllm-ascend-dev-image
export CONTAINER_NAME=vllm-ascend-dev
export DEVICE=/dev/davinci1

# 首次构建会花费10分钟（10MB/s）下载基础镜像和包
docker build -t $IMAGE -f ./Dockerfile .
# 您还可以通过设置 VLLM_REPO 来指定镜像仓库以加速
# docker build -t $IMAGE -f ./Dockerfile . --build-arg VLLM_REPO=https://gitee.com/mirrors/vllm

docker run --rm --name $CONTAINER_NAME --network host --device $DEVICE \
           --device /dev/davinci_manager --device /dev/devmm_svm \
           --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi \
           -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
           -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
           -ti $IMAGE bash

cd vllm-ascend
pip install -r requirements-dev.txt

pytest tests/
```

## 开发者来源证书(DCO)

在向本项目提交贡献时，您必须同意 DCO。提交必须包含“Signed-off-by:”标头，以证明同意 DCO 的条款。

在`git commit`时使用`-s`将会自动添加该标头。

## PR 标题和分类

仅特定类型的 PR 会被审核。PR 标题会以适当的前缀来表明变更类型。请使用以下之一：

- `[Attention]` 关于`attention`的新特性或优化
- `[Communicator]` 关于`communicators`的新特性或优化
- `[ModelRunner]` 关于`model runner`的新特性或优化
- `[Platform]` 关于`platform`的新特性或优化
- `[Worker]` 关于`worker`的新特性或优化
- `[Core]` 关于`vllm-ascend`核心逻辑 (如 `platform, attention, communicators, model runner`)的新特性或优化
- `[Kernel]` 影响计算内核和操作的更改.
- `[Bugfix]` bug修复
- `[Doc]` 文档的修复与更新
- `[Test]` 测试 (如：单元测试)
- `[CI]` 构建或持续集成改进
- `[Misc]` 适用于更改内容对于上述类别均不适用的PR，请谨慎使用该前缀

> [!注意]
> 如果 PR 涉及多个类别，请添加所有相关前缀

## 其他

您可以在 [<u>docs.vllm.ai</u>](https://docs.vllm.ai/en/latest/contributing/overview.html) 上找到更多有关为 vLLM 昇腾插件贡献的信息。
如果您在贡献过程中发现任何问题，您可以随时提交 PR 来改进文档以帮助其他开发人员。
