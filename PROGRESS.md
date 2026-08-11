# OKFolio 实验进度面板

更新于: 2026-08-11

## 当前阶段
实验数据同步 GitHub（v3-v6 运行、探针、覆盖草稿、工具，含脱敏与快照重算）已完成；Phase 4（修复运行）冻结，等待授权 v7。

## 运行进度
v6 正式运行已完成 [████████████████████] 332/332 处理

## 关键数字
accepted 140 / withheld 192 / failed 0 / 剩余 0 / ETA 无（v6 已结束，v7 待授权）

## 下一步
补 47 覆盖草稿 → 授权并启动 v7 修复运行（`experiment-data/runs/v7-launcher/run_v7.sh` 已就绪，待补覆盖后直接执行）。

## 里程碑
- 2026-08-10 17:13 探针 v13（oversized 批式探针）完成
- 2026-08-10 17:26 探针 v14（fixed）完成
- 2026-08-10 18:07 v4（正式批式 claim review）运行
- 2026-08-10 19:59 v5（remote qwen3p6）运行
- 2026-08-11 16:05 v6（remote 无思考）运行结束：332/332 处理、140 accepted、192 withheld
- 2026-08-11 覆盖草稿 /tmp/overrides 77 份就绪，缺陷工作清单生成
- 2026-08-11 实验数据同步 GitHub：脱敏（MinIO host → minio.internal）+ 6 份 source_snapshot 哈希重算自检通过

## 运行进程
空闲（服务器无后台运行；v7 未授权未启动）
