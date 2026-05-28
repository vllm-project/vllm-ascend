#!/bin/bash

if [ -d "/proc/ascend_ub" ]; then
    cd /proc/ascend_ub
else
    cd /proc/asdrv_ub
fi

ls dev_id

# 定义输出文件
OUTPUT_FILE="/lib/route.conf"

# 动态获取davinci设备编号（排除davinci_manager）
DEVICES=($(awk '/^ns_id/ {ns_count++} ns_count==1 && /^[[:space:]]*[0-9]+/{print $2}' /proc/uda/namespace_node | sort -n))

# 检查是否找到设备
if [[ ${#DEVICES[@]} -eq 0 ]]; then
    echo "错误: 未找到任何davinci设备"
    exit 1
fi

if [ ! -e "/lib/route.conf.bak" ]; then
    cp -r "$OUTPUT_FILE" "/lib/route.conf.bak"
fi

# 判断输出文件是否存在，如果存在就退出
if [ -e "$OUTPUT_FILE" ]; then
    rm -f $OUTPUT_FILE
    echo "已删除旧配置文件:  $OUTPUT_FILE 已存在，准备重新生成"
    # exit 1
fi

# 清空或创建输出文件
> "$OUTPUT_FILE"

# 写入pair_device_num
echo "pair_device_num=${#DEVICES[@]}" >> "$OUTPUT_FILE"
echo "找到 ${#DEVICES[@]} 个davinci设备: ${DEVICES[@]}"

# if [ -e "/home/source_dest_ips.txt" ]; then
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     python3 $SCRIPT_DIR/get_route_eid.py
#     exit 0
# fi

# 检查必要的文件是否存在
if [[ ! -f "dev_id" ]]; then
    echo "错误: dev_id 文件不存在"
    exit 1
fi

if [[ ! -f "pair_info" ]]; then
    echo "错误: pair_info 文件不存在"
    exit 1
fi

INDEX=0

echo "开始处理设备..."

# 分析设备ID的范围和分组
MIN_DEVICE=${DEVICES[0]}
MAX_DEVICE=${DEVICES[${#DEVICES[@]}-1]}
echo "设备ID范围: $MIN_DEVICE - $MAX_DEVICE"

# 计算每个设备的组内索引（相对于8的倍数）
for i in "${!DEVICES[@]}"; do
    num=${DEVICES[i]}
    
    # 计算当前设备在8个一组中的位置
    GROUP_OFFSET=$((num % 8))
    
    # 根据在组内的位置决定使用哪组eid
    # 前4个设备（0-3）使用第一组eid，后4个设备（4-7）使用第二组eid
    if [ $GROUP_OFFSET -lt 4 ]; then
        EID_SELECTOR="head -1"
        EID_GROUP="第一组(组内位置: $GROUP_OFFSET)"
    else
        EID_SELECTOR="tail -1" 
        EID_GROUP="第二组(组内位置: $GROUP_OFFSET)"
    fi
    
    echo "正在处理设备 davinci$num (全局索引: $i, 组内位置: $GROUP_OFFSET, 使用: $EID_GROUP)..."
    
    # 将设备编号写入dev_id
    echo "$num" > dev_id
    
    # 检查写入是否成功
    if [[ $? -ne 0 ]]; then
        echo "错误: 无法写入 dev_id 文件"
        continue
    fi
    
    # 等待一下确保信息更新
    sleep 0.1
    
    # 获取pair_info内容
    PAIR_INFO=$(cat pair_info 2>/dev/null)
    if [[ $? -ne 0 ]]; then
        echo "错误: 无法读取 pair_info 文件"
        continue
    fi
    
    # 提取slot_id（从pair_info中解析）
    SLOT_ID=$(echo "$PAIR_INFO" | grep "dev_id=$num" | sed -n 's/.*slot_id=\([0-9]*\).*/\1/p')
    if [[ -z "$SLOT_ID" ]]; then
        echo "警告: 无法从pair_info中提取slot_id，使用设备编号 $num 作为slot_id"
        SLOT_ID=$num
    fi
    
    # 根据组内位置选择对应的eid组
    LOCAL_EID=$(echo "$PAIR_INFO" | grep "local_eid:" | $EID_SELECTOR | awk '{print $2}' | sed 's/://g')
    REMOTE_EID=$(echo "$PAIR_INFO" | grep "remote_eid:" | $EID_SELECTOR | awk '{print $2}' | sed 's/://g')
    
    # 检查是否成功提取到EID
    if [[ -z "$LOCAL_EID" || -z "$REMOTE_EID" ]]; then
        echo "警告: 无法从pair_info中提取$EID_GROUP EID信息，尝试使用备选方案..."
        
        # 如果当前选择的eid组不存在，尝试使用另一组
        if [ $GROUP_OFFSET -lt 4 ]; then
            # 原本应该用第一组，但不存在，尝试第二组
            LOCAL_EID=$(echo "$PAIR_INFO" | grep "local_eid:" | tail -1 | awk '{print $2}' | sed 's/://g')
            REMOTE_EID=$(echo "$PAIR_INFO" | grep "remote_eid:" | tail -1 | awk '{print $2}' | sed 's/://g')
            EID_GROUP="第二组(备选)"
        else
            # 原本应该用第二组，但不存在，尝试第一组
            LOCAL_EID=$(echo "$PAIR_INFO" | grep "local_eid:" | head -1 | awk '{print $2}' | sed 's/://g')
            REMOTE_EID=$(echo "$PAIR_INFO" | grep "remote_eid:" | head -1 | awk '{print $2}' | sed 's/://g')
            EID_GROUP="第一组(备选)"
        fi
        
        # 再次检查
        if [[ -z "$LOCAL_EID" || -z "$REMOTE_EID" ]]; then
            echo "错误: 无法提取任何EID信息，跳过设备 $num"
            continue
        fi
    fi
    
    # 格式化EID，确保长度正确并添加0x前缀
    LOCAL_EID="0x$(printf "%032s" "$LOCAL_EID" | sed 's/ /0/g')"
    REMOTE_EID="0x$(printf "%032s" "$REMOTE_EID" | sed 's/ /0/g')"
    
    # 写入route.conf文件
    echo "pair${INDEX}_dev_id=$((num % 8))" >> "$OUTPUT_FILE"
    echo "pair${INDEX}_slot_id=$SLOT_ID" >> "$OUTPUT_FILE"
    echo "pair${INDEX}_chan_num=1" >> "$OUTPUT_FILE"
    echo "pair${INDEX}_chan0_local_eid=$LOCAL_EID" >> "$OUTPUT_FILE"
    echo "pair${INDEX}_chan0_remote_eid=$REMOTE_EID" >> "$OUTPUT_FILE"
    echo "pair${INDEX}_chan0_flag=1" >> "$OUTPUT_FILE"
    
    echo "设备 davinci$num 处理完成 -> pair$INDEX ($EID_GROUP)"
    
    # 索引递增
    ((INDEX++))
    
    # 添加空行分隔（可选）
    if [[ $INDEX -lt ${#DEVICES[@]} ]]; then
        echo "" >> "$OUTPUT_FILE"
    fi
done

echo "路由配置文件生成完成: $OUTPUT_FILE"
echo "共处理了 $INDEX 个设备"

# 显示生成的文件内容
echo -e "\n生成的route.conf文件内容:"
echo "=========================================="
cat "$OUTPUT_FILE"
echo "=========================================="
