# ARC3 Agent Infra

**单卡离线大模型推理 · 多会话 Agent 运行时 · 自动化评测**

我为 ARC-AGI-3 构建的长程 Agent 推理系统：在一张 RTX PRO 6000 上运行量化大模型，让多个游戏会话共享推理服务，并完成“观察 → 推理 → Python 工具执行 → 环境反馈”的持续交互。

项目重点是把有限的显存、推理时间和上下文预算，转化为更多有效的游戏探索与解题动作。

## 本地评测表现

**46.39 平均分 · 7/25 游戏通关 · 603.78 token/s · 94.24% 缓存 token 复用**

| 指标 | 本地测评结果 |
|---|---:|
| 评测集 | ARC-AGI-3 Public25，离线运行，1 pass |
| 平均得分 | **46.3908** |
| 完整通关 | **7 / 25（28%）** |
| 完成关卡 | **104 / 183** |
| 生成吞吐（按作业墙钟时间） | **603.78 token/s** |
| Actor 缓存 token 复用比例 | **94.24%** |
| 生成 tokens（作业汇总） | **4,425,369** |
| 游戏运行耗时 | **2小时2分9秒** |

数据来自项目的完整本地实验 **357355935**。当前模块化源码从 **356766364** 拆分；实验日志、计算口径和版本对应关系见 [评测记录](docs/benchmarks.md)。

## 使用了什么技术，解决了什么问题

| 目标 | 技术实现 | 系统效果 |
|---|---|---|
| **单卡容纳模型与长上下文** | W4A16 权重量化 + FP8 KV Cache | 降低权重与 KV 存储开销；单张 RTX PRO 6000 部署共享模型服务 |
| **提高生成效率** | NEXTN/MTP 投机解码 + CUDA Graph | 批量验证候选 token，减少逐步解码与 kernel launch 开销；完整系统本地吞吐 **603.78 token/s** |
| **减少重复预填充** | Radix 前缀缓存 + LPM 调度 | 复用 Agent 连续请求的公共前缀；Actor 缓存 token 占比 **94.24%** |
| **承载长程交互** | 128K Agent 窗口 + 上下文 drain | 支持较长动作历史，在上下文达到阈值时整理历史，控制输入增长 |
| **控制共享推理负载** | 游戏会话准入 + 10路活跃流 + Chunked Prefill | 限制同时争用 GPU 的请求；分块处理长输入 |
| **提升执行可靠性** | Function Repair + game 级函数持久化 + No-op Recovery | 修复工具函数问题，隔离会话状态，在重复无效动作时触发局部恢复 |
| **让优化可以复盘** | 独立实验 worker + 请求/游戏遥测 + 资源监控 | 自动生成逐游戏结果、错误统计、配置快照与实验组对照 |

表中的吞吐和缓存比例是整套系统的测量值；单项技术收益通过后续消融比较。

## 工程结构

| 目录 / 模块 | 职责 |
|---|---|
| `src/arc3/serving.py` | 推理引擎安装、模型检查与服务启动 |
| `src/arc3/environment.py` | 离线环境、策略配置与 harness 接入 |
| `src/arc3/benchmark.py` | 游戏评测与正式运行入口 |
| `src/arc3/experiment.py` | 实验调度、遥测、结果汇总 |
| `src/arc3/monitor.py` | CPU / 内存 / 显存 / 磁盘监控 |
| `src/arc3/assets/` | Harness patch 与局部恢复逻辑 |
| `configs/` | 可编辑的运行与推理配置 |
| `benchmarks/` | 本地评测汇总与结果数据 |
| `tests/` | 配置、预算、入口路由与清理回归测试 |
| `notebooks/kaggle_entry.ipynb` | Kaggle 轻量启动入口 |

原始 `arc3_2459.ipynb` 保留作历史快照。后续开发直接修改 `src/` 和 `configs/`。

## 快速开始

先在普通 CPU 环境检查配置与运行测试：

```bash
PYTHONPATH=src python -m arc3 --config configs/baseline.json --check
PYTHONPATH=src python -m unittest discover -s tests -v
```

在 Kaggle RTX PRO 6000 环境附加模型、wheelhouse、source bundle 和比赛输入后运行：

```bash
PYTHONPATH=src python -m arc3 --config configs/baseline.json
```

默认执行 **lf52 / 5分钟 / 1路活跃流** 的 smoke test。完整评测、挂载路径与输出说明见 [运行指南](docs/running.md)。

## 文档

- [本地评测数据与统计口径](docs/benchmarks.md)
- [运行指南与配置](docs/running.md)
- [系统架构与依赖](docs/architecture.md)

**作者：陈守杰 · Shoujie Chen**
