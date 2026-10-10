查看npu设备

```
npu-smi info
```



模型启动

```
vllm serve ./Qwen3-VL-8B-Instruct --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16
384
```



接口验证

```
curl -X POST http://127.0.0.1:8000/v1/chat/completions \-H "Content-Type: application/json" \-d '{
"messages" : [ { "role" : "user", "content" : "介绍一下自己" } ],
"temperature" : 0,
"max_tokens": 256,
"models": "Qwen3-VL-8B-Instruct",
"seed":42
}'

{"id":"chatcmpl-88e4ff0cb3a486a0","object":"chat.completion","created":1784700127,"model":"./Qwen3-VL-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"你好！我是通义千问（Qwen），由阿里巴巴集团旗下的通义实验室自主研发的超大规模语言模型。我的中文名是通义千问，英文名是Qwen。我能够回答问题、创作文字，比如写故事、写公文、写邮件、写剧本、逻辑推理、编程等，还能表达观点，玩游戏等。我支持多种语言，包括但不限于中文、英文、德语、法语、西班牙语等。\n\n我的训练数据截止于2024年，因此我能够提供较为丰富和最新的信息。我可以在不同场景下提供帮助，无论是学术研究、工作协助还是日常生活中的问题，我都会尽力为你提供支持。\n\n我有多个版本，包括Qwen、Qwen2、Qwen3等，每个版本在性能和功能上都有所不同，以适应不同的使用需求。例如，Qwen3是最新版本，具有更强的推理能力和更广泛的知识覆盖。\n\n如果你有任何问题或需要帮助，欢迎随时告诉我，我会尽我所能为你提供帮助！","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":10,"total_tokens":227,"completion_tokens":217,"prompt_tokens_details":null,"completion_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

性能测试：

```
vllm bench serve   --base-url http://localhost:8000   --model Qwen3-VL-8B-Instruct --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1
```

