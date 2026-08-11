# OKFolio 实验进度面板

更新于: 2026-08-11

## 当前阶段
单根目录整合与全量重命名（kmpro* → OKFolio）已完成并推送；**端到端恢复演练通过**（全新克隆可继续实验）；实验数据（v3-v6 运行、探针、覆盖草稿、工具）同步 GitHub 完成；Phase 4（修复运行 v7）**已暂缓**（用户明确 V7 后置）。

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
- 恢复演练（/tmp 全新克隆端到端）[████████████████████] 4/4
  - 克隆落地：source-run 9 文件、structures 10、runs v3-v6（v6 checkpoints 332）、overrides 77、v7-launcher ✓
  - 哈希比对：v6 source_snapshot.json 的 inputs+structures 13/13 全等 ✓
  - 无模型 resume：配置相等 + 快照校验在任何模型调用前通过；exit 1 "did not pass every group"；llm.done 0 次新增 ✓
  - 缺口修复 3 处（见里程碑）：快照比对布局无关化、launcher 自动物化 data/、legacy 配置归一化；提交 `44f2509`、`73068aa` 均已推送且 CI 全绿

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
1. [已暂缓] 补 47 覆盖草稿（用户明确 V7 后置）
2. [已暂缓] v7 修复运行（同前；`experiment-data/runs/v7-launcher/run_v7.sh` 已就绪，首次运行会自动从 `experiment-data/` 物化 `data/`）
3. 3090 恢复指引：clone → `pip install -r requirements.lock pytest pytest-httpx` → 导出 API 环境变量 → 续跑冻结 run 直接用 `experiment-data/` 路径 `--resume`（快照校验布局无关）；或执行 `run_v7.sh`（自动物化 data/）

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
- 2026-08-11 端到端恢复演练（`/tmp/okfolio-restore-drill` 全新克隆）：13/13 快照哈希全等；无模型 resume 校验全通（llm.done 0 新增）；修复 3 个缺口
  - `44f2509`：快照比对改为哈希承载、与目录名无关；v7 launcher 首次运行自动物化 `data/`；HANDOVER/experiment-data-README 更新恢复配方
  - `73068aa`：legacy 配置归一化（`draft_overrides` null→{}、seed 记录补齐缺省 prompt-relaxed 标志）
  - 新增回归测试：重定位布局 resume、legacy 配置形态 resume（409 passed）

## 运行进程
空闲（服务器无后台运行；v7 未授权未启动）
