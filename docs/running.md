# 运行指南

## 环境

GPU运行面向 Kaggle Linux / RTX PRO 6000。推理引擎使用离线 wheelhouse 中的 Pennyroyal，需 Python 3.12；可通过 `ARC3_INFERENCE_PYTHON` 指定兼容解释器。CPU上的配置检查与测试无需GPU或模型。

附加五项输入：比赛 ARC Prize 2026 - ARC-AGI-3、`pennyroyal-v253`、`TAAF Kaggle Source Bundle (copy)`、Intel Qwen3.8-Flash-Next W4A16 AutoRound v1、Albucino Qwen3.8-Flash-Next drafter v1。

将仓库作为 Kaggle Dataset 挂载，仓库文件放在该 Dataset 根目录。打开 `notebooks/kaggle_entry.ipynb`，按实际路径修改 `PROJECT`。入口将源码加入 `sys.path`，加载配置并调用 `await run(settings)`。

命令行运行不依赖 Notebook：

```bash
PYTHONPATH=src python -m arc3 --config configs/baseline.json --check
PYTHONPATH=src python -m arc3 --config configs/baseline.json
```

`--check`仅验证配置，不加载模型或检查输入挂载。实际运行先验证输入目录。

## 配置

修改 `configs/baseline.json`：

| 配置区 | 常用项 |
|---|---|
| `paths` | `model`、`draft_model`、`wheelhouse`、`source_bundle`、`working` |
| `run` | `DEV_MODE`、`DEV_GAME_PREFIXES`、`DEV_MINUTES_PER_ARM`、`DEV_ACTIVE_STREAMS` |
| `inference` | `CTX`、`CHUNK`、`MAXREQ`、`MEMFRAC`、`SPEC`、`KVDTYPE` |

默认 `smoke`：lf52单游戏、5分钟、1次、1路活跃流。`compare`与`confirm`由实验运行器组织各profile；`full`运行完整benchmark。配置中的profile、预算、图捕获容量和prefill块大小会在启动前检查。

`bundle_work`使用专用可写目录 `/kaggle/taaf-kaggle-source-share`。每次运行重建该副本并应用patch；源输入目录保持只读。服务端口沿用8001。

## 正式比赛

Kaggle的 `KAGGLE_IS_COMPETITION_RERUN` 决定是否走正式gateway与正式预算。保留原版9小时总预算及原有优先级调度覆盖逻辑。不要手动伪造该变量进行本地评测。

开发模式的 `submission.parquet` 是Notebook输出占位文件；比赛运行通过gateway产生正式结果。

## 输出

- `serve.log`：服务启动和推理日志。
- `experiments/<run>/comparison.csv`、`per_game.csv`：开发实验与逐游戏结果。
- 各实验组的 `telemetry.jsonl`、`effective_config.json`、`run.log`：事件、配置和日志。
- 完整运行的 `run_metrics/` 与harness诊断文件：执行轨迹与结果。

进程退出时停止监控线程，并终止本次启动的模型服务进程组。Notebook历史快照可以独立运行，但后续项目开发以 `src/arc3` 为主。
