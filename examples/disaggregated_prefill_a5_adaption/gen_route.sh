#!/bin/bash

if [ -d "/proc/ascend_ub" ]; then
    cd /proc/ascend_ub
else
    cd /proc/asdrv_ub
fi

ls dev_id

# Define the output file.
OUTPUT_FILE="/lib/route.conf"

# Dynamically collect Davinci device IDs, excluding davinci_manager.
DEVICES=($(awk '/^ns_id/ {ns_count++} ns_count==1 && /^[[:space:]]*[0-9]+/{print $2}' /proc/uda/namespace_node | sort -n))

# Verify that devices were found.
if [[ ${#DEVICES[@]} -eq 0 ]]; then
    echo "Error: no Davinci devices found."
    exit 1
fi

if [ ! -e "/lib/route.conf.bak" ]; then
    cp -r "$OUTPUT_FILE" "/lib/route.conf.bak"
fi

# Remove the existing output file before regenerating it.
if [ -e "$OUTPUT_FILE" ]; then
    rm -f "$OUTPUT_FILE"
    echo "Removed existing route configuration file: $OUTPUT_FILE. Regenerating it."
    # exit 1
fi

# Clear or create the output file.
> "$OUTPUT_FILE"

# Write pair_device_num.
echo "pair_device_num=${#DEVICES[@]}" >> "$OUTPUT_FILE"
echo "Found ${#DEVICES[@]} Davinci devices: ${DEVICES[@]}"

# if [ -e "/home/source_dest_ips.txt" ]; then
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     python3 $SCRIPT_DIR/get_route_eid.py
#     exit 0
# fi

# Verify that required files exist.
if [[ ! -f "dev_id" ]]; then
    echo "Error: dev_id file does not exist."
    exit 1
fi

if [[ ! -f "pair_info" ]]; then
    echo "Error: pair_info file does not exist."
    exit 1
fi

INDEX=0

echo "Starting device processing..."

# Analyze the device ID range and grouping.
MIN_DEVICE=${DEVICES[0]}
MAX_DEVICE=${DEVICES[${#DEVICES[@]}-1]}
echo "Device ID range: $MIN_DEVICE - $MAX_DEVICE"

# Calculate each device's offset within its group, relative to a multiple of 8.
for i in "${!DEVICES[@]}"; do
    num=${DEVICES[i]}

    # Calculate the current device's position within an 8-device group.
    GROUP_OFFSET=$((num % 8))

    # Select the EID group based on the position within the group.
    # The first four devices (0-3) use the first EID group, and the last
    # four devices (4-7) use the second EID group.
    if [ $GROUP_OFFSET -lt 4 ]; then
        EID_SELECTOR="head -1"
        EID_GROUP="first group, group offset: $GROUP_OFFSET"
    else
        EID_SELECTOR="tail -1"
        EID_GROUP="second group, group offset: $GROUP_OFFSET"
    fi

    echo "Processing device davinci$num (global index: $i, group offset: $GROUP_OFFSET, using: $EID_GROUP)..."

    # Write the device ID to dev_id.
    echo "$num" > dev_id

    # Verify that the write succeeded.
    if [[ $? -ne 0 ]]; then
        echo "Error: failed to write dev_id file."
        continue
    fi

    # Wait briefly to ensure the device information is refreshed.
    sleep 0.1

    # Read pair_info content.
    PAIR_INFO=$(cat pair_info 2>/dev/null)
    if [[ $? -ne 0 ]]; then
        echo "Error: failed to read pair_info file."
        continue
    fi

    # Extract slot_id from pair_info.
    SLOT_ID=$(echo "$PAIR_INFO" | grep "dev_id=$num" | sed -n 's/.*slot_id=\([0-9]*\).*/\1/p')
    if [[ -z "$SLOT_ID" ]]; then
        echo "Warning: failed to extract slot_id from pair_info. Using device ID $num as slot_id."
        SLOT_ID=$num
    fi

    # Select the corresponding EID group based on the group offset.
    LOCAL_EID=$(echo "$PAIR_INFO" | grep "local_eid:" | $EID_SELECTOR | awk '{print $2}' | sed 's/://g')
    REMOTE_EID=$(echo "$PAIR_INFO" | grep "remote_eid:" | $EID_SELECTOR | awk '{print $2}' | sed 's/://g')

    # Verify that EID values were extracted successfully.
    if [[ -z "$LOCAL_EID" || -z "$REMOTE_EID" ]]; then
        echo "Warning: failed to extract EID information for $EID_GROUP from pair_info. Trying the fallback EID group..."

        # If the selected EID group does not exist, try the other group.
        if [ $GROUP_OFFSET -lt 4 ]; then
            # The first group was expected but is unavailable. Try the second group.
            LOCAL_EID=$(echo "$PAIR_INFO" | grep "local_eid:" | tail -1 | awk '{print $2}' | sed 's/://g')
            REMOTE_EID=$(echo "$PAIR_INFO" | grep "remote_eid:" | tail -1 | awk '{print $2}' | sed 's/://g')
            EID_GROUP="second group, fallback"
        else
            # The second group was expected but is unavailable. Try the first group.
            LOCAL_EID=$(echo "$PAIR_INFO" | grep "local_eid:" | head -1 | awk '{print $2}' | sed 's/://g')
            REMOTE_EID=$(echo "$PAIR_INFO" | grep "remote_eid:" | head -1 | awk '{print $2}' | sed 's/://g')
            EID_GROUP="first group, fallback"
        fi

        # Verify again.
        if [[ -z "$LOCAL_EID" || -z "$REMOTE_EID" ]]; then
            echo "Error: failed to extract any EID information. Skipping device $num."
            continue
        fi
    fi

    # Format EID values to ensure the correct length and add the 0x prefix.
    LOCAL_EID="0x$(printf "%032s" "$LOCAL_EID" | sed 's/ /0/g')"
    REMOTE_EID="0x$(printf "%032s" "$REMOTE_EID" | sed 's/ /0/g')"

    # Write route.conf entries.
    echo "pair${INDEX}_dev_id=$((num % 8))" >> "$OUTPUT_FILE"
    echo "pair${INDEX}_slot_id=$SLOT_ID" >> "$OUTPUT_FILE"
    echo "pair${INDEX}_chan_num=1" >> "$OUTPUT_FILE"
    echo "pair${INDEX}_chan0_local_eid=$LOCAL_EID" >> "$OUTPUT_FILE"
    echo "pair${INDEX}_chan0_remote_eid=$REMOTE_EID" >> "$OUTPUT_FILE"
    echo "pair${INDEX}_chan0_flag=1" >> "$OUTPUT_FILE"

    echo "Device davinci$num processed successfully -> pair$INDEX ($EID_GROUP)"

    # Increment the pair index.
    ((INDEX++))

    # Add a blank line between entries.
    if [[ $INDEX -lt ${#DEVICES[@]} ]]; then
        echo "" >> "$OUTPUT_FILE"
    fi
done

echo "Route configuration file generated successfully: $OUTPUT_FILE"
echo "Processed $INDEX devices."

# Display the generated file content.
echo -e "\nGenerated route.conf content:"
echo "=========================================="
cat "$OUTPUT_FILE"
echo "=========================================="
