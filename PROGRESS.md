# OKFolio 实验进度面板

更新于: 2026-08-11

## 当前阶段
单根目录整合与全量重命名（kmpro* → OKFolio）已完成并推送；实验数据（v3-v6 运行、探针、覆盖草稿、工具）同步 GitHub 完成；Phase 4（修复运行）冻结，等待授权 v7。

## 当前任务（单根重构/重命名）进度

- 任务 A：单根 data/ 整合 [████████████████████] 4/4
  - 数据迁移：`~/kmpro-wiki-v15-data` → `data/`（agent-runs/、normalized-sources/、sources/、processed/、parser-jobs/、corpus-run.json、normalization-report.json）；`~/区域经济皮书` → `data/pdfs/`（10 PDF）
  - 符号链接删除：`.local-runtime/`（agent-runs、normalized-sources 两条链接）已确认并整目录删除
  - 路径验证：manifest.json 存在、structures 10/10、pdfs 10/10、`review_concept_claims.py --help` exit 0、默认 structures_dir 解析为 `data/normalized-sources` ✓
  - 引用与 .gitignore：RUNBOOK/HANDOVER/EXPERIMENT_STATUS/experiment-data-README/run_v7.sh 已改 `data/...`；.gitignore 移除 `.local-runtime/`、新增 `data/`
  - 提交：`72e290f`（7 files，+55/-44）已推送；GitHub Actions 全绿
- 任务 B：kmpro* → OKFolio 全量重命名 [████████████████████] 4/4
  - 包重命名：`git mv kmpro_wiki okfolio`（65 个文件 rename）
  - 引用替换：147 files（+273/-260）：imports、Dockerfile、docs、examples、MCP scheme `kmpro://` → `okfolio://`、资源显示名、临时前缀
  - 数据兼容：schema 写入字面量改 `okfolio.*`；旧数据读取保留 legacy 字面量（见下）
  - 验证：必测 4 文件 132 passed；全量 407 passed / 2 skipped；`audit_open_source.py` exit 0；`import okfolio.agentwiki.claim_review` OK
  - 提交：`61db8db` 已推送；GitHub Actions 全绿

### 保留的 legacy 字面量（读取旧数据/旧品牌，勿改）
- 代码读取端：`kmpro.document-structure.v1`（agentic.py、corpus.py 两处，注释说明兼容）
- 旧品牌脱敏映射：`"KMPro Wiki": "OKFolio"`（build_public_demo.py，测试断言 renamed_legacy_brand_values=1）
- 测试 fixture：`kmpro.document-structure.v1` / `kmpro.page-result.v1`（9 处，模拟旧持久化格式）
- 数据与历史记录：`kmpro.agent-run.v2` 等 schema 字符串（data/ 与 experiment-data/ 内字节一致数据）、MinIO bucket `kmpro-wiki-assets`、迁移来源路径 `kmpro-wiki-v15-data`、根 `MANIFEST.sha256`（机器生成快照文物，未动）

## 运行进度
v6 正式运行已完成 [████████████████████] 332/332 处理

## 关键数字
accepted 140 / withheld 192 / failed 0 / 剩余 0 / ETA 无（v6 已结束，v7 待授权）；覆盖草稿 77/124 [████████████████░░░░]

## 下一步
1. 补 47 覆盖草稿（至 124）
2. 授权并启动 v7 修复运行：`experiment-data/runs/v7-launcher/run_v7.sh`（已改指向 `$ROOT/data/...`，待补覆盖后直接执行）
3. 3090 恢复：clone 后按 HANDOVER 恢复 `data/`（从 `experiment-data/` 快照或本机 master）

## 里程碑
- 2026-08-10 17:13 探针 v13（oversized 批式探针）完成
- 2026-08-10 17:26 探针 v14（fixed）完成
- 2026-08-10 18:07 v4（正式批式 claim review）运行
- 2026-08-10 19:59 v5（remote 运行）开始
- 2026-08-11 16:05 v6（remote 无思考）运行结束：332/332 处理、140 accepted、192 withheld
- 2026-08-11 覆盖草稿 /tmp/overrides 77 份就绪，缺陷工作清单生成
- 2026-08-11 实验数据同步 GitHub：脱敏（MinIO host → minio.internal）+ 6 份 source_snapshot 哈希重算自检通过
- 2026-08-11 单根目录整合：master 数据迁入仓库根 `data/`（含 `data/pdfs/` 10 本原始 PDF），删除 `.local-runtime/` 符号链接机制；提交 `72e290f`
- 2026-08-11 全量重命名 kmpro* → OKFolio：包 `kmpro_wiki` → `okfolio`，schema 读写保持旧数据兼容；提交 `61db8db`
- 2026-08-11 GitHub Actions 两次 push 均全绿（72e290f / 61db8db）

## 运行进程
空闲（服务器无后台运行；v7 未授权未启动）
