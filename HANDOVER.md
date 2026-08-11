# HANDOVER — 本机收尾与 3090 接手

更新于: 2026-08-11

## 1. 本机权威数据位置（master）

本机（当前工作站）的权威数据在 `.local-runtime/`，**它不进 git**（.gitignore 已忽略）：

- `.local-runtime/agent-runs/` — 全部 agent 运行目录；其中 v2-source、v3、v12、v1、v2 是指向原始只读工作区
  `~/Documents/Codex/2026-07-15/zh/kmpro-wiki-v15-system-experiment-20260729/agent-runs/` 的符号链接；
  v4/v5/v6/v13/v14 是实体目录。
- `.local-runtime/normalized-sources/` — 指向同一原始工作区的 `normalized-sources/`（符号链接），内含 10 个
  `*.structure.json`。

**GitHub 是副本**：克隆后如需继续实验，一律以 GitHub 上 `experiment-data/` 为数据源，或以本机
`.local-runtime/` 重建（两者字节一致，见下节）。

## 2. GitHub 恢复映射

| GitHub `experiment-data/` | 本机 `.local-runtime/` |
| --- | --- |
| `source-run/` | `agent-runs/public10-local-qwen36-semantic-v2-20260809` |
| `structures/` | `normalized-sources/*.structure.json`（10 个） |
| `runs/public10-claim-review-formal-qwen36-v3-20260810` | `agent-runs/…-v3-…` |
| `runs/public10-claim-review-formal-qwen36-v4-batched-20260810` | `agent-runs/…-v4-batched-…` |
| `runs/…-remote-v5-…` | `agent-runs/…-v5-…` |
| `runs/…-remote-v6-nothinking-…` | `agent-runs/…-v6-nothinking-…` |
| `probes/…-v13-batched-…`、`probes/…-v14-fixed-…` | `agent-runs/…-v13-batched-…`、`agent-runs/…-v14-fixed-…` |
| `overrides/` | `/tmp/overrides`（已删除，见清理清单；77 个覆盖草稿已入库） |
| `tools/extract_defects.py`、`tools/defects_worklist_20260811.md` | `/tmp/extract_defects.py`、`/tmp/defects_worklist.md`（已删除，已入库） |

复制方式为 `cp -RL`（符号链接已解引用）；GitHub 副本与原始资产字节一致（含内网 MinIO 资产端点原样保留——用户判定内容为公开 PDF 的 MinerU 解析产物，不涉密，原样入库保证可恢复性；端点地址见 experiment-data/README.md 的敏感度决策记录）。

## 3. 当前实验状态

- v6 正式运行（`experiment-data/runs/` 下的 v6 目录，remote 无思考）：**完成**
  332/332 组处理、**140 accepted / 192 withheld**（manifest: status=partial, completed=264, reviews=192）。
- **Phase 4（修复运行 v7）冻结**：等待授权；后续移 3090 执行。
- 覆盖草稿：**77/124 完成，缺 47**。
- 探针 v13/v14 已完成（批式覆盖探针与修复探针）。

## 4. 3090 接手步骤

1. `git clone git@github.com:Eijnewgnaw/OKFolio.git`（实验数据已在 `experiment-data/`）。
2. 安装依赖：`pip install -r requirements.lock`（如需 RAG 评估再加 `requirements-rag.lock`）。
3. 本地 vLLM 服务模型（模型名以环境变量注入）：
   `export OPENAI_BASE_URL=…`、`export OPENAI_MODEL=<v6 启动时所用模型名>`、`export OPENAI_API_KEY=…`、`export NO_PROXY=…`。
   （参考部署文档 `docs/research/2026-08-11-3090-local-deployment.md`——截至 2026-08-11 **尚未产出**，
   产出后按文档执行。）
4. 继续实验：
   - 续跑冻结 run：`python3 scripts/review_concept_claims.py --source-run experiment-data/source-run
     --output-dir experiment-data/runs/<frozen-run> --resume …`（v6 快照哈希未变，`_verify_snapshot` 通过）。
   - 启动 v7 修复运行：`experiment-data/runs/v7-launcher/run_v7.sh`（路径相对仓库根解析，API 走环境变量；
     `--draft-override-dir experiment-data/overrides`；先补齐 47 个覆盖草稿）。
5. 进度自查：仓库根 `PROGRESS.md`（每次状态变更由执行者更新）。

## 5. 本机收尾建议（推荐，未执行）

- LM Studio 本地服务可停用：管线已不再使用本地 LM Studio 端点（v5/v6 走远程网关，v7 走 3090 本地 vLLM）。
- 本机 `.local-runtime/` 保留作为 master 归档，勿删。

## 6. 清理清单（2026-08-11 已执行）

已删除（内容均已字节一致性验证后入库或可再生）：
- `/tmp/overrides/`（77 文件，336K）→ 已入库 `experiment-data/overrides/`（逐文件 cmp 一致）
- `/tmp/extract_defects.py`、`/tmp/defects_worklist.md` → 已入库 `experiment-data/tools/`（cmp 一致）
- `/tmp/run_v7.sh` → 仓库已有脱敏版 `experiment-data/runs/v7-launcher/run_v7.sh`
- `/tmp/sections/`（420K，可再生中间产物）
- 运行日志与 PID：`/tmp/v4-run*.log`、`/tmp/v5-run.log`、`/tmp/v6-run.log`、`/tmp/v4.pid`、`/tmp/v5.pid`、`/tmp/v6.pid`
- 并发/探针脚本与结果：`/tmp/lm_concurrency_bench.log`、`lm_concurrency_bench.py`、`lm_concurrency_bench_results.json`、`lms_probe.py`、`lms_probe2.py`
- 校验基线：`/tmp/a.sorted`、`b.sorted`、`s1`、`s2`、`v1s`、`v2s`、`s2b`、`probe_baseline.txt`、
  `probe_events_before_resume.log`、`probe_files_before_resume.sha256`、`v3_baseline.sha256`、`v3_check.sha256`、
  `v3_after.sha256`、`source_baseline.sha256`、`source_check.sha256`、`source_after.sha256`、
  `v14_before_resume.sha256`、`v14_after_resume.sha256`
- 本机快照缓存：`.pytest_cache/`、全部 `__pycache__/`（可再生）

未触碰：`.local-runtime/`、`~/.ssh/`、仓库代码与数据、用户环境变量配置（含 DT 前缀项）。
