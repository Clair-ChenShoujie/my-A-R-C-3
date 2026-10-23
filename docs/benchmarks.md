# 本地评测记录

## 完整 Public25 实验

结果来源：[notebook5fa958a77c / 357355935](https://www.kaggle.com/code/clairechenxyz/notebook5fa958a77c?scriptVersionId=357355935) 的已完成离线运行输出。

- [原始最终汇总](../benchmarks/public25_summary.txt)
- [机器可读结果](../benchmarks/public25_results.json)
- 原 Notebook SHA256：`c4ad65bae6269837d2cdcbe6bb311c820fcb35ba969728fac4a9c49a8ce904e8`
- 完整25题、1 pass；平均分46.39084599027885、7题通关、104关完成。
- 时间：日志原始时区下 2026-10-11 13:20:44 至 15:22:53，耗时7329秒。

## 推理指标怎么算

**生成吞吐**使用作业汇总的 `generated tokens/sec = 603.78`，以整个游戏作业的墙钟时间为分母。对应 `total tokens = 4,425,369`。它反映闭环 Agent 工作负载，包含工具执行、等待等开销；并非单请求 decode 速度。

**缓存 token 复用比例**使用 Actor 请求级统计：

```text
prompt_tokens = 177,449,853
cached_tokens = 167,219,712
cached / prompt = 94.2349%
uncached_prompt_tokens = 10,230,141
```

这是 token 加权比例，不是请求命中率。Actor 请求级输出 tokens 为4,517,717，与作业汇总口径分别保存。并发请求的 latency 总和及逐游戏 wallclock 总和都不作为作业耗时。

## 代码与实验版本

| 版本 | 用途 |
|---|---|
| `356766364` | 本仓库模块化源码的来源；原始代码快照为根目录 Notebook |
| `357355935` | README 展示的项目历史完整 Public25 实验 |
| 本次工程化 | 模块拆分、JSON配置、CLI、轻量Notebook入口与CPU回归；未启动GPU重测 |

这些历史结果用于展示项目已经完成的本地实验。新的模块化版本生成独立的日志和配置快照，之后可继续在本目录记录对照结果。

## 建议的后续对照

固定请求轨迹，对比 MTP on/off、不同 prefill chunk 和活跃流上限；记录 TTFT、TPOT、P95排队时间、生成吞吐、显存与缓存比例。再用闭环 Public25 比较平均分及通关数。
