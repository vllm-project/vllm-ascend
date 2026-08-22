import requests
import time

# 配置区（按需修改）
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3.5:latest"  # 确保模型名与本地拉取的一致
REQUEST_TIMEOUT = 120  # 延长超时时间至120秒，解决长文本生成超时问题

# 测试用例（8个用例完整保留，命中逻辑已统一改为关键词匹配）
test_cases = [
    {
        "name": "基础常识问答",
        "input": "请简要说明光合作用的基本过程。",
        "keywords": ["光伏作用", "暗反应", "二氧化碳", "氧气", "葡萄糖"],
        "total": 5
    },
    {
        "name": "逻辑推理题",
        "input": "A 比 B 高，B 比 C 高，D 比 C 矮。请从高到矮排序。",
        "keywords": ["B > C"], 
        "total": 1
    },
    {
        "name": "代码生成",
        "input": "写一个Python函数，输入列表返回其中的偶数。",
        "keywords": ["def", "even", "% 2 == 0", "append", "return"],
        "total": 5
    },
    {
        "name": "数学计算",
        "input": "计算 1+2+3+4+5+6+7+8+9+10 的结果，直接给答案。",
        "keywords": ["55"],
        "total": 1
    },
    {
        "name": "长文本摘要",
        "input": "人工智能是模拟人类智能的技术，近年因大模型与算力突破快速发展。请总结。",
        "keywords": ["人工智能", "大模型", "技术"],
        "total": 3
    },
    {
        "name": "多轮一致性",
        "input": "我喜欢编程、健身、阅读。我平时喜欢做什么？",
        "keywords": ["编程", "健身", "阅读"],
        "total": 3
    },
    {
        "name": "指令遵循",
        "input": "用三句话介绍北京，每句不超过10字。",
        "keywords": ["北京"],
        "total": 3
    },
    {
        "name": "翻译任务",
        "input": "将这句话翻译成法语：The Great Wall has been a symbolic representation of China's endurance against invasions for ages.",
        "keywords": ["La Grande Muraille", "Chinois"],
        "total": 2
    }
]

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