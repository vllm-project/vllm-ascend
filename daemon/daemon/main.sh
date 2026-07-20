#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <source_file>"
    exit 1
fi

SOURCE_FILE=$1
OUTPUT_FILE=$2

HANDLE_POOL_FILE="$(pwd)/handle_pool.cpp"
MODEL_MGMT_FILE="$(pwd)/model_management.cpp"
DAEMON_FILE="$(pwd)/daemon.cpp"

g++ "$SOURCE_FILE" $HANDLE_POOL_FILE $MODEL_MGMT_FILE $DAEMON_FILE -std=c++17 \
    -I /usr/local/Ascend/ascend-toolkit/latest/include \
    -I $(pwd)/inc \
    -L /usr/local/Ascend/ascend-toolkit/latest/lib64 \
    -lascendcl \
    -o "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Compilation successful. Output file: $OUTPUT_FILE"
else
    echo "Compilation failed!"
    exit 1
fi