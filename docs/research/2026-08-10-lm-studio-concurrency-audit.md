# LM Studio 本地 MLX 服务并发能力审计（qwen3.6-35b-a3b-mlx）

日期：2026-08-10  
核验对象：LM Studio 0.4.20+1（本机 `lms` CLI commit `71bd99c`），本地服务 `http://localhost:1234/v1`，模型 `qwen3.6-35b-a3b-mlx`（MLX，Apple M3 Max / 36 GB）。  
边界：纯客户端只读测量。不改动仓库任何文件，不触碰 `.local-runtime/agent-runs` 数据，不运行 claim review，不调用任何外部 API；脚本与原始结果都放在 `/tmp`，不在 MANIFEST 追踪范围内。

## 环境配置

- base URL：`http://localhost:1234`（host:port）。来源顺序：仓库根 `.env`（不存在）→ shell 环境变量 `OPENAI_BASE_URL`（未设置）→ `.env.example`（只有模板，值为空）→ LM Studio 默认端口。无 API key：本地服务不要求（RUNBOOK 确认），请求不携带 Authorization 头。
- model：`qwen3.6-35b-a3b-mlx`，与 `.local-runtime/agent-runs/*/manifest.json` 的 `configuration.client.stages.*.model` 一致；`GET /v1/models` 实际列出该 id。
- 固定 prompt：`用一句话说明知识库检索的作用。`，`temperature=0`。
- 请求体对齐生产管线（`okfolio/agentwiki/llm.py`）：`stream=true`、`stream_options.include_usage=true`、`chat_template_kwargs={"enable_thinking": false}`；SSE 流式解析与计时复用 `okfolio/evaluation/llm_benchmark.py` 的做法。
- 脚本：`/tmp/lm_concurrency_bench.py`；原始结果：`/tmp/lm_concurrency_bench_results.json`；运行日志：`/tmp/lm_concurrency_bench.log`。全程 148 秒，未触发 20 分钟上限（剩余 1052s）。

## 环境行为观察（先行探针）

1. **`enable_thinking=false` 未生效**：即使发送 `chat_template_kwargs={"enable_thinking": false}`，模型仍先输出 570 个 `reasoning_content` token，之后才输出 30 个 content token（`usage.completion_tokens_details.reasoning_tokens=570`）。这与 2026-08-09 文档的预警一致：「服务接受参数」不等于「模型完全不产生 reasoning」。
2. **输出确定且等长**：每个请求恰好生成 600 completion tokens（570 reasoning + 30 content，54 个 content 字符），`finish_reason=stop`，全部 20 个请求输出逐字节一致（temperature=0）。
3. **单请求节奏**：TTFT（首个 reasoning token）≈ 0.12–0.15s（warmup 轮 2.4s，属冷启动）；content 在 ≈ 6.97s 才开始；稳态 ≈ 82.5 tok/s；单请求墙钟 ≈ 7.28s。
4. **服务端 FIFO 串行**：并发轮中第 2 个请求的首个 token 恰好出现在第 1 个请求结束时刻（例如并发 2 第 1 轮：7.35s ≈ 7.24s），无交错执行、无抢占。

## 方法

- 串行基线：`max_tokens=8192`，1 次 warmup + 3 次正式请求，取 3 次正式请求均值（tok/s 取每请求 tok/s 的均值）。
- 并发 2 / 3 / 4：线程同时发起 N 个 `max_tokens=8192` 流式请求，等全部完成。并发 2、3 各 2 轮，并发 4 1 轮（轮数预算用上一轮观测的每请求延迟估算；该延迟含服务端排队，导致估算偏高，并发 4 只排了 1 轮，属规格允许的 1–2 轮）。
- 可选长输出：并发 2 × `max_tokens=16384`，1 轮，观察是否 OOM/换页。
- 每轮记录：wall time（首请求发出到末请求结束）、聚合 tok/s（Σ completion_tokens ÷ wall）、每请求延迟（含服务端排队）、`finish_reason`、usage tokens、错误。
- 判定基准：串行单请求均值 7.28s。round wall ÷ 7.28 ≈ 1 → 真并行；≈ 并发数 → 串行排队。

## 数据

| 场景（并发 × max_tokens） | 轮数 | wall time (s) | 聚合 tok/s | 每请求延迟 mean (s) | wall ÷ 串行单请求 | 吞吐增益 |
|---|---:|---:|---:|---:|---:|---:|
| 1 × 8192（串行基线） | 3 | 7.26 / 7.26 / 7.31，均值 **7.28** | 82.6 / 82.7 / 82.1，均值 **82.5** | 7.28 | 1.00 | 1.00 |
| 2 × 8192 | 2 | 14.50 / 14.56，均值 **14.53** | 82.7 / 82.4，均值 82.6 | 10.9 | 1.99 / 2.00 | 1.00 / 1.00 |
| 3 × 8192 | 2 | 21.82 / 21.93，均值 **21.88** | 82.5 / 82.1，均值 82.3 | 14.6 | 3.00 / 3.01 | 1.00 / 1.00 |
| 4 × 8192 | 1 | **29.25** | 82.1 | 18.3 | **4.02** | 1.00 |
| 2 × 16384 | 1 | **14.58** | 82.3 | 10.9 | 2.00 | 1.00 |

每请求延迟明细（并发轮，≈ 7.3s × 服务端排队位置）：

- 并发 2：`[7.2, 14.5]`；并发 3：`[7.3, 14.5, 21.8]`；并发 4：`[7.3, 14.6, 21.9, 29.2]`

错误：**0**。20 个请求全部 `finish_reason=stop`，无超时、无 4xx/5xx；`max_tokens=16384` 轮与 8192 轮行为完全一致（仍 600 token 自然停），无 OOM、无换页迹象。

## 结论

1. **该 MLX 后端严格串行，无任何并发吞吐收益**：2 路 wall = 1.99–2.00 × 串行单请求，3 路 = 3.00–3.01 ×，4 路 = 4.02 ×；聚合 tok/s 恒定 ≈ 82，并发 2/3/4 的吞吐增益均为 1.00×（最高 1.003×），**任何并发路数都无法达到 ≥1.5× 吞吐提升**。
2. **服务端排队本身安全**：4 路并发零错误、无 OOM；代价是每请求延迟随排队位置线性增长（≈ 7.3s × 位置），即并发只是把同样的总工作量按顺序排完。
3. **推荐并发因子：1（保持串行）**。当前管线「阶段内单请求、阶段间串行」的用法已经是该服务的最优用法；客户端并发只会增加队列等待，不提高吞吐。若应用因架构需要并发发出请求，服务端会安全排队（实测 4 路无错），但不要指望吞吐提升。
4. **如需 ≥1.5× 吞吐，应换后端/运行时而非提高客户端并发**：例如支持连续批处理的 vLLM、SGLang，或多实例负载均衡；在单实例 LM Studio MLX 后端上提高并发数没有意义。

## 复现

```bash
python3 /tmp/lm_concurrency_bench.py   # 输出 /tmp/lm_concurrency_bench_results.json
```

只读探测依据：`GET http://localhost:1234/v1/models`（列出 `qwen3.6-35b-a3b-mlx` 等 5 个模型）、`lms ls --json`、`lms server status`、`lms --version`、RUNBOOK.md（本地服务无需 API key）、`.local-runtime/agent-runs/*/manifest.json`（模型 id 与阶段配置：`send_chat_template_kwargs=true`、`enable_thinking=false`、`max_tokens=8192`）。
