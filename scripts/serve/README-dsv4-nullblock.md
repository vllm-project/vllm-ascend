# DeepSeek V4 压缩 KV null-block A/B（绿区执行）

本分支的源码**已经打了修复**。B 臂（已修）直接跑；A 臂（未修）用同目录的 patch 反向撤销。

配套文件：

- `scripts/serve/serve_dsv4_dspark.sh` —— 普通单进程启动脚本（DP4 + EP，DSpark 5，FP8 KV，`FULL_DECODE_ONLY`）
- `scripts/serve/dsv4-null-block.patch` —— 本次修复的完整 diff，可用 `git apply -R` 回退

## 0. 确认改动落在被 import 的那份源码上

```bash
PKG=$(python3 -c 'import pathlib, vllm_ascend; print(pathlib.Path(vllm_ascend.__file__).resolve().parent)')
echo "$PKG"
grep -c in_null_block "$PKG/models/deepseek_v4/compressor.py"     # 1 = 已修, 0 = 未修
python3 -c 'import json;print(json.load(open("/mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8/config.json")).get("dspark_block_size"))'   # 必须是 5
```

## 1. B 臂（已修，预期正常停止）

```bash
nohup bash scripts/serve/serve_dsv4_dspark.sh > /tmp/serve_B.log 2>&1 &
# 等日志出现 Uvicorn running / 或轮询
until curl -sf http://127.0.0.1:18000/health >/dev/null; do sleep 10; done

PROMPT='<和原调查同一条 105-token prompt>'
for i in 1 2 3; do
  curl -sS http://127.0.0.1:18000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-v4\",\"prompt\":\"$PROMPT\",\"max_tokens\":192,\"temperature\":0}" \
    > /tmp/probe-B-$i.json
  python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));c=d["choices"][0];print(sys.argv[1],"tokens=",d["usage"]["completion_tokens"],"finish=",c["finish_reason"],"chars=",len(c["text"]),repr(c["text"][:60]))' /tmp/probe-B-$i.json
done

pkill -TERM -f 'vllm serve'
```

## 2. A 臂（未修，预期只输出 BOS 直到长度上限）

```bash
git apply -R scripts/serve/dsv4-null-block.patch
grep -c in_null_block "$PKG/models/deepseek_v4/compressor.py"     # 0

nohup bash scripts/serve/serve_dsv4_dspark.sh > /tmp/serve_A.log 2>&1 &
until curl -sf http://127.0.0.1:18000/health >/dev/null; do sleep 10; done
# 同上再打 3 次，存 /tmp/probe-A-$i.json
pkill -TERM -f 'vllm serve'
```

## 3. 恢复（或直接 `git checkout -- .`）

```bash
git apply scripts/serve/dsv4-null-block.patch
```

## 4. 判定

| A 臂 | B 臂 | 结论 |
|---|---|---|
| 3 次都 `finish_reason=length`、192 tokens、文本空或近空 | 正常 `stop` 且有内容 | 复现且修复生效 |
| 正常 | 正常 | 该配置未进入故障路径 —— 不是"修复无效" |
| 坏 | 坏 | 先核对两臂 `grep -c in_null_block` 输出确实不同 |

记录要留下的东西：两臂各自的 `grep -c` 结果、`git log -1 --format=%H`、checkpoint 路径与 `dspark_block_size`、caputre 配置、6 条 probe 的 `tokens/finish_reason/chars`，以及日志里 `grep -i -e nan -e inf` 的结果。

注意：`max_tokens=192` 与 `max_num_seqs(32) × 6` 是同一个数字只是巧合，写结论时分开。
