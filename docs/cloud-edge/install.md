# 安装指南

## 1 环境要求
- Atlas A2/A3 服务器 或 一体机 
- 操作系统：openeuler
- Python：>= 3.10, < 3.13
- 配备昇腾 NPU 的硬件，通常为 Atlas 800 A2 系列。
- docker软件
- 
软件：

| 软件       | 支持版本                   | 说明                                                          |
|------------|----------------------------|---------------------------------------------------------------|
| Ascend HDK | 请参阅 CANN 9.1.0 版本说明 | CANN 所需                                                     |
| CANN       | == 9.1.0                   | vllm-ascend 和 TorchNPU 所必需                                |
| TorchNPU   | == 2.10.0.post4            | vllm-ascend 所需，无需手动安装，将在以下步骤中自动安装        |
| torch      | == 2.10.0                  | TorchNPU 和 vllm 所必需，无需手动安装，将在以下步骤中自动安装 |
| NNAL       | == 9.1.0                   | libatb.so 所需，支持高级张量运算                              |

## 2 操作系统环境安装
### 2.1 服务器
操作系统安装参考：

[Atlas服务器openEuler 22.03 LTS SP4 操作系统安装](https://support.huawei.com/enterprise/zh/doc/EDOC1100469526/426cffd9?idPath=23710424|269761314|269769397|269769917|254184887)

### 2.2 一体机
操作系统安装参考：

[Atlas服务器openEuler 24.03 LTS SPX 操作系统安装](https://support.huawei.com/enterprise/zh/doc/EDOC1100492838/426cffd9?idPath=23710424|269761314|269769397|269769917|261457531)


## 3 昇腾配套驱动环境安装
### 3.1 服务器
NPU驱动和固件参考：

[Atlas A2 中心推理和训练硬件 26.0.RCx NPU驱动和固件](https://support.huawei.com/enterprise/zh/doc/EDOC1100568431/426cffd9?idPath=23710424|269761314|269769397|269769917|254184887)
### 3.2 一体机
NPU驱动和固件安装参考：

[Atlas A2 中心推理和训练硬件 26.0.RCx NPU驱动和固件](https://support.huawei.com/enterprise/zh/doc/EDOC1100568431/426cffd9?idPath=23710424|269761314|269769397|269769917|254184887)

SP681网卡驱动安装参考：

[SP220&SP600 标准网卡 用户指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100573093/426cffd9)

[驱动下载地址](https://support.huawei.com/enterprise/zh/computing-module/in220-pid-253287505/software/268628573?idAbsPath=fixnode01|23710424|269761301|269764423|269766398|253287505)


## 4 节点通信验证
首先检查物理层链接，然后验证节点间链接
### 4.1 物理层要求
- 服务器NPU均通过光模块链接到交换机，且连接状态正常
- 一体机NPU无需连线，但是roce网卡需要通过光模块连接到CPE设备，且连接状态正常
### 4.2 互连验证
(1) 获取IP

获取服务器NPU IP地址
```commandline
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done 
```
获取一体机roce网卡ip
```commandline
ip addr show
```

(2) 跨节点PING测试

从服务器ping
```commandline
hccn_tool -i 0 -ping -g address x.x.x.x 
```
从一体机ping
```commandline
ping x.x.x.x
```

## 5 使用docker软件设置环境
我们提供用户部署的Docker镜像，您可以从[镜像仓库](https://quay.io/repository/kyoshiki/vllm-ascend-v23.0rc1?tab=tags)拉取进行，并在节点运行以下示例命令启动容器
- A2
```commandline
export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0rc1
docker run -itd --privileged \
--name vllm-ascend \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```
- A3
```commandline
export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0.rc1-a3
docker run -itd --privileged \
--name vllm-ascend \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

