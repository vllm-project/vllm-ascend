import requests
import time
import yaml
import os

# 配置区（按需修改）
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3.5:latest"  # 确保模型名与本地拉取的一致
REQUEST_TIMEOUT = 120  # 延长超时时间至120秒，解决长文本生成超时问题

# ===================== 自动加载测试用例（从 yaml 读取）=====================
# 自动找到 configs 下的测试用例文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(SCRIPT_DIR, "configs", "phi3-5-moe-instruct-test-cases.yaml")

with open(YAML_PATH, "r", encoding="utf-8") as f:
    test_cases = yaml.safe_load(f)["test_cases"]
# =========================================================================

def main():
    print(f"开始测试模型：{MODEL_NAME}")
    total_all = 0
    hit_all = 0

    for case in test_cases:
        print(f"\n=== {case['name']} ===")
        print(f"输入：{case['input']}")

        full_prompt = case['input']
        total = case["total"]
        total_all += total

        # 带超时和异常捕获的请求逻辑
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"num_ctx": 4096}  # 增大上下文窗口
                },
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()  # 如果请求失败（非200状态码），抛出异常

            answer = response.json().get("response", "").strip()
            print(f"回答：{answer}")

            # 核心修复：统一使用关键词匹配逻辑，彻底移除check_func
            hit = sum(1 for kw in case["keywords"] if kw in answer)
        
            hit_all += hit
            print(f"命中：{hit}/{total} ({hit/total*100:.1f}%)")

        except requests.exceptions.Timeout:
            print(f"=== {case['name']} 失败 ===")
            print("错误：请求超时（120秒内无响应）")
            continue
        except Exception as e:
            print(f"=== {case['name']} 失败 ===")
            print(f"错误：{str(e)}")
            continue

    print(f"\n=== 测试总结 ===")
    print(f"总用例数：{len(test_cases)}")
    print(f"总命中：{hit_all}/{total_all} ({hit_all/total_all*100:.1f}%)")

if __name__ == "__main__":
    main()