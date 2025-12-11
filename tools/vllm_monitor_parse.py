import re
import pandas as pd

PARSE_RULES = [
    {
        'pattern': r".*?P: (\d+), D: (\d+), chunked_P: (\d+), recomputed: (\d+), P_tokens: (\d+), chunked_P_tokens: (\d+), recomputed_tokens: (\d+), computed: (\d+\.\d+), kv_cache_usage: (\d+\.\d+)",
        'fields': ['P', 'D', 'chunked_P', 'recomputed', 'P_tokens', 'chunked_P_tokens', 'recomputed_tokens', 'computed', 'kv_cache_usage'],
        'post_processor': lambda x: [*map(int, x[:-2]), int(float(x[-2])), float(x[-1])]
    },
    {
        'pattern': r"visual_input: (\[.*?\]), visual_output: (\[.*?\]), vit_forward: (\d+\.\d+)",
        'fields': ['visual_input', 'visual_output', 'vit_forward'],
        'post_processor': lambda x: [x[0], x[1], float(x[2])]
    },
    {
        'pattern': r"prepare_data: (\d+\.\d+)",
        'fields': ['prepare_data'],
        'post_processor': lambda x: [float(x[0])]
    },
    {
        'pattern': r"aclgraph: CUDAGraphMode.([\w]+)",
        'fields': ['aclgraph'],
        'post_processor': lambda x: [x[0]]
    },
    {
        'pattern': r"forward: (\d+\.\d+)",
        'fields': ['forward'],
        'post_processor': lambda x: [float(x[0])]
    },
    {
        'pattern': r"post_process: (\d+\.\d+)",
        'fields': ['post_process'],
        'post_processor': lambda x: [float(x[0])]
    },
    #{
    #    'pattern': r"kv_cache_usage: (\d+\.\d+)",
    #    'fields': ['kv_cache_usage'],
    #    'post_processor': lambda x: [float(x[0])]
    #}
]

# --- 2. 全局配置 ---
INPUT_PATH = "monitor_tgs_3000.log"
OUTPUT_PATH = "monitor_tgs_3000.csv"
NUM_RANKS = [0,4,8,12]  # 从 world_size 更名为 NUM_RANKS，更清晰

def parse_log_for_rank(rank, log_lines, parse_rules):
    # 初始化最终数据存储字典
    save_dict = {field:[] for rule in parse_rules for field in rule['fields']}
    # 临时存储一轮匹配的数据（用于校验是否完整匹配）
    temp_round_data = {field: None for field in save_dict.keys()}

    # 筛选出当前 rank 的相关日志行
    rank_lines = [line for line in log_lines if f"========rank: {rank}" in line]
    current_rule_index = 0  # 状态：当前期望匹配的规则索引
    num_rules = len(parse_rules)
    is_dummy_run = False
    has_vit_P = False

    for line in rank_lines:
        print(f"{current_rule_index}/{num_rules}, {line}")
        # 检测dummy_run标记
        if f"dummy_run" in line:
            is_dummy_run = True
            # dummy_run时先给临时数据填充0（非kv字段）
            for field in temp_round_data:
                temp_round_data[field] = 0

        # 处理visual_input规则：仅当P有值时才解析，否则填充0
        if current_rule_index == 1 and not has_vit_P:
            for field in parse_rules[current_rule_index]['fields']:
                temp_round_data[field] = 0
            current_rule_index +=1

        current_rule = parse_rules[current_rule_index]
        current_fields = current_rule['fields']


        # 正常模式：解析规则并填充临时数据
        if match := re.search(current_rule['pattern'], line):
            is_dummy_run = False
            values = list(match.groups())
            # 应用后处理函数
            if 'post_processor' in current_rule:
                try:
                    values = current_rule['post_processor'](values)
                except (ValueError, TypeError) as e:
                    print(f"警告: 在 rank {rank} 处理规则 '{current_fields[0]}' 时转换失败: {values}。错误: {e}")
            # 将解析值存入临时数据
            for field, value in zip(current_fields, values):
                temp_round_data[field] = value

            # 标记P是否有值（用于控制visual_input规则）
            if current_fields[0] == 'P':
                has_vit_P = temp_round_data['P'] > 0
                print(f"****************************has_vit_P {has_vit_P}")

            current_rule_index += 1

        # 如果所有规则都匹配过一轮：校验完整性并保存，然后重置状态
        if current_rule_index >= num_rules or is_dummy_run or (has_vit_P and current_rule_index == num_rules):
            # 检查是否所有字段都有值（一轮完整匹配）
            if all(v is not None for v in temp_round_data.values()):
                # 完整匹配，将临时数据存入最终字典
                for field in save_dict:
                    save_dict[field].append(temp_round_data[field])
            print(f"%%%%%%%%%%%% temp_round_data {temp_round_data}")
            # 重置临时数据和状态，准备下一轮
            temp_round_data = {field: None for field in save_dict.keys()}
            current_rule_index = 0
            has_vit_P = False
            is_dummy_run = False
            continue

    # 处理最后一轮未完成的匹配（日志结束时）
    if all(v is not None for v in temp_round_data.values()):
        for field in save_dict:
            save_dict[field].append(temp_round_data[field])

    return save_dict

def main():
    print(f"开始解析日志: {INPUT_PATH}")

    # 一次性读取所有日志行
    with open(INPUT_PATH, "r", errors='ignore') as f:
        all_lines = f.readlines()

    all_dfs = []
    for rank in NUM_RANKS:
        print(f"正在解析 Rank {rank}...")
        parsed_data = parse_log_for_rank(rank, all_lines, PARSE_RULES)

        # 转换为 DataFrame 并添加 rank 后缀
        df = pd.DataFrame(parsed_data)
        df = df.add_suffix(f"_{rank}")
        all_dfs.append(df)

    # 合并所有 rank 的 DataFrame
    if all_dfs:
        df_total = pd.concat(all_dfs, axis=1)
        # 保存到 CSV 文件
        df_total.to_csv(OUTPUT_PATH, index=False)
        print(f"解析完成！结果已保存到: {OUTPUT_PATH}")
        print(f"共解析到 {len(df_total)} 行有效数据")
    else:
        print("警告: 没有生成任何 DataFrame，可能没有解析到任何数据。")

if __name__ == "__main__":
    main()
