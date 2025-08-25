#!/bin/bash
IMAGE_NAME="quay.nju.edu.cn/ascend/cann:8.2.rc1-a3-ubuntu22.04-py3.11"
sudo docker pull $IMAGE_NAME

CONTAINER_NAME="ascend_ci_a3"

if docker ps -a --format '{{.Names}}' | grep -qw "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' exists. Removing it..."

    if docker ps --format '{{.Names}}' | grep -qw "^${CONTAINER_NAME}$"; then
        echo "Stopping container '$CONTAINER_NAME'..."
        docker stop "$CONTAINER_NAME"
    fi

    docker rm "$CONTAINER_NAME"
    echo "Container '$CONTAINER_NAME' has been removed."
fi

echo "starting ascend NPU-A3 container"

# Run the container using the defined variables
docker run -itd \
    --name "$CONTAINER_NAME" \
    --net=host \
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
    -e CLUSTER_SIZE:$CLUSTER_SIZE \
    -e MASTER_ADDR:$MASTER_ADDR \
    -e WORKSPACE="/home/workspace" \
    -v $GITHUB_WORKSPACE:/home/workspace \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/actions-runner/.cache:/root/actions-runner/.cache \
    -v /mnt/sfs_turbo/ascend-ci-share-nv-action-vllm-benchmarks:/root/.cache \
    $IMAGE_NAME bash

# Check if container start successfully
if [ $? -eq 0 ]; then
  echo "Container $CONTAINER_NAME start successfully"
else
  echo "Container $CONTAINER_NAME start failed, please check if the images exist or permission"
  exit 1
fi
