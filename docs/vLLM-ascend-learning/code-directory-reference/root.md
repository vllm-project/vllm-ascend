# 根目录文件：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **39 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 仓库根目录

仓库根目录：安装构建、容器、开发规范、依赖与项目介绍。

| 子目录 | 职责 |
| --- | --- |
| [.agents/](agents.md) | 开发辅助技能与操作说明，不参与模型前向执行。 |
| [.claude/](claude.md) | 开发助手相关配置或说明，不属于推理运行代码。 |
| [.gemini/](gemini.md) | 开发助手相关配置或说明，不属于推理运行代码。 |
| [.github/](github.md) | GitHub 协作与自动化：CI、工作流、Issue/PR 模板和版本参考。 |
| [benchmarks/](benchmarks.md) | 性能测量脚本、输入配置和报告模板。 |
| [cmake/](cmake.md) | 仓库顶层 CMake 构建辅助。 |
| [csrc/](csrc.md) | 自定义设备算子、PyTorch/CANN 绑定及算子构建工具链。 |
| [docs/](docs.md) | 项目文档、站点构建、主题覆盖与当前学习资料。 |
| [examples/](examples.md) | 离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [tests/](tests.md) | 单元、系统、集成、性能和测试基础设施；不是服务运行入口。 |
| [tools/](tools.md) | 开发检查、调试、请求发送、文档生成与性能工具。 |
| [vllm_ascend/](vllm_ascend.md) | Ascend 插件 Python 主包：将上游 vLLM 的引擎和模型接口接到 NPU 实现。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [.clang-format](../../../.clang-format) | C/C++ 格式化规则。 | 依据文件格式及所在目录归类。 |
| [.gitignore](../../../.gitignore) | 规定哪些本地产物不进入版本控制。 | 依据文件格式及所在目录归类。 |
| [.gitleaks.toml](../../../.gitleaks.toml) | 敏感信息扫描的规则配置。 | 配置条目与规则；按实际读取者生效。 |
| [.gitmodules](../../../.gitmodules) | 声明第三方 Git 子模块地址与路径。 | 依据文件格式及所在目录归类。 |
| [.markdownlint.yaml](../../../.markdownlint.yaml) | Markdown 格式规则。 | 顶层项：`MD007`, `MD013`, `MD024`, `MD031`, `MD029` |
| [.pre-commit-config.yaml](../../../.pre-commit-config.yaml) | 配置格式、静态检查、拼写和其他提交/CI 检查钩子。 | 顶层项：`default_install_hook_types`, `default_stages`, `exclude`, `repos` |
| [.readthedocs.yaml](../../../.readthedocs.yaml) | Read the Docs 文档构建环境与任务配置。 | 顶层项：`version`, `build`, `mkdocs`, `formats` |
| [AGENTS.md](../../../AGENTS.md) | 本仓开发助手需遵守的实现、测试、NPU 与提交规则。 | 标题：vLLM Ascend Development Guidelines |
| [CLAUDE.md](../../../CLAUDE.md) | 开发助手相关指引，非运行期代码。 | 文档正文；用途按所属主题归类。 |
| [CMakeLists.txt](../../../CMakeLists.txt) | 顶层 CMake 构建入口，组织 C++/NPU 扩展与依赖。 | 构建项：`vllm_ascend_C` |
| [CODE_OF_CONDUCT.md](../../../CODE_OF_CONDUCT.md) | 社区行为规范。 | 标题：vLLM Code of Conduct |
| [CONTRIBUTING.md](../../../CONTRIBUTING.md) | 贡献流程与开发协作入口。 | 标题：Contributing to vLLM Ascend |
| [DCO](../../../DCO) | 贡献者签署贡献来源声明的规则。 | 依据文件格式及所在目录归类。 |
| [Dockerfile](../../../Dockerfile) | 构建 通用 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.310p](../../../Dockerfile.310p) | 构建 310p 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.310p.openEuler](../../../Dockerfile.310p.openEuler) | 构建 310p.openEuler 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.a3](../../../Dockerfile.a3) | 构建 a3 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.a3.openEuler](../../../Dockerfile.a3.openEuler) | 构建 a3.openEuler 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.a5](../../../Dockerfile.a5) | 构建 a5 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.a5.openEuler](../../../Dockerfile.a5.openEuler) | 构建 a5.openEuler 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.openEuler](../../../Dockerfile.openEuler) | 构建 openEuler 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [LICENSE](../../../LICENSE) | 项目许可证文本。 | 依据文件格式及所在目录归类。 |
| [README.md](../../../README.md) | 项目英文总介绍、安装/使用入口和相关链接。 | 标题：Overview |
| [README.zh.md](../../../README.zh.md) | 项目中文总介绍、安装/使用入口和相关链接。 | 标题：总览 |
| [cleanup-main-branches.ps1](../../../cleanup-main-branches.ps1)（本地未跟踪） | 此前生成的本地分支清理操作脚本；当前未跟踪，运行前需核对目标。 | 依据文件格式及所在目录归类。 |
| [codecov.yml](../../../codecov.yml) | 测试覆盖率报告与检查配置。 | 顶层项：`coverage` |
| [collect_env.py](../../../collect_env.py) | 收集 Python、框架、设备和运行环境信息，供问题诊断使用。 | 定义：`run`, `run_and_read_all`, `run_and_parse_first_match` |
| [commit-vllm-cann-adaptation.ps1](../../../commit-vllm-cann-adaptation.ps1)（本地未跟踪） | 此前生成的本地适配提交脚本，供用户在有 .git 写权限时执行。 | 依据文件格式及所在目录归类。 |
| [dsnext-merged-pr-inventory.json](../../../dsnext-merged-pr-inventory.json)（本地未跟踪） | 此前 GitCode 已合并 PR 扫描记录：来源提交与变更文件等。 | JSON list 数据。 |
| [format.sh](../../../format.sh) | 调用 pre-commit，执行开发或 CI 阶段的格式/静态检查。 | 函数：`check_command` |
| [mkdocs.yml](../../../mkdocs.yml) | 文档站点的导航、主题和插件配置。 | 顶层项：`site_name`, `site_url`, `site_description`, `site_author`, `repo_url` |
| [mypy.ini](../../../mypy.ini) | Python 类型检查范围与规则。 | 配置条目与规则；按实际读取者生效。 |
| [packages.txt](../../../packages.txt) | 环境构建需要的系统软件包清单。 | 依据文件格式及所在目录归类。 |
| [pyproject.toml](../../../pyproject.toml) | 定义 Python 构建与开发工具的项目级配置。 | 配置条目与规则；按实际读取者生效。 |
| [requirements-dev.txt](../../../requirements-dev.txt) | 开发环境依赖清单。 | 依据文件格式及所在目录归类。 |
| [requirements-lint.txt](../../../requirements-lint.txt) | 格式化与静态检查工具依赖清单。 | 依据文件格式及所在目录归类。 |
| [requirements.txt](../../../requirements.txt) | 运行依赖清单。 | 依据文件格式及所在目录归类。 |
| [setup.py](../../../setup.py) | 定义 Python 包构建、NPU 扩展编译和 vLLM 插件 entry points。 | 定义：`CMakeExtension`, `custom_develop`, `load_module_from_path`, `check_or_set_default_env`, `get_value_from_lines` |
| [typos.toml](../../../typos.toml) | 拼写检查配置与允许词。 | 配置条目与规则；按实际读取者生效。 |
