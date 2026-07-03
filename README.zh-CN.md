# autoresearch-for-ascend

[English](README.md) | [中文](README.zh-CN.md)

这是一个把 Andrej Karpathy `autoresearch` 思想迁移到昇腾/NPU 环境上的实验仓库。

这个仓库保留的核心思想很简单：

1. 固定数据和评估代码；
2. 只允许智能体改实验配置或实验文件；
3. 每次实验都在固定预算下运行；
4. 指标变好就保留；
5. 指标没变好就放弃。

当前原型面向华为昇腾 910C 服务器，运行在隔离 Docker 容器中，使用本地
Qwen3-0.6B 权重。

## 项目来源

本项目来自对几个仓库和框架的研究：

- `karpathy/autoresearch`：最小化的自动研究循环。
- `Ascend/MindSpeed-MM`：昇腾生态参考代码和环境参考。
- `Ascend/MindSpeed-LLM`：当前 Qwen3-0.6B 加速框架实验使用的训练栈。

这里的实现不是上述任何仓库的直接 fork，而是一个面向我们现有硬件、网络限制
和模型条件的小型昇腾适配原型。

外部参考仓库会放在本地 `reference/` 目录下用于学习，但不会提交到本仓库。

## 当前结构

```text
ascend_autoresearch/
  README.md       本地使用说明
  prepare.py      固定的数据加载、tokenize 和验证逻辑
  train.py        可编辑的实验文件
  program.md      智能体实验协议
  run_train.sh    昇腾环境启动脚本
framework_adapters/
  mindspeed_llm/  MindSpeed-LLM smoke/评估脚本
    candidates/   版本化候选 env 配置
docs/
  framework_evaluation.md
  updates/        每个版本的更新说明
CHANGELOG.md      版本历史
```

## 运行环境假设

已验证的远端运行环境：

- 容器名：`llin-autoresearch`
- 容器工作区：`/workspace/ascend_autoresearch`
- 模型路径：`/models/Qwen3-0.6B`
- 默认设备范围：一个可见昇腾 NPU

容器只暴露了较窄的挂载和设备面：项目工作区可写，模型目录只读，只暴露一个
NPU 设备，避免影响物理机和其他容器。

## 版本历史

| 版本 | 日期 | 摘要 |
| --- | --- | --- |
| v0.3.10 | 2026-07-03 | 用 6-step 探测 `9.0e-5` 边界；新的最佳 raw HF val_loss 是 `12.420206`。 |
| v0.3.9 | 2026-07-03 | 用 6-step 探测 `8.0e-5` 边界；新的最佳 raw HF val_loss 是 `12.434883`。 |
| v0.3.8 | 2026-07-03 | 在 `4.5e-5` 细化 12-step 边界，并用 6-step 探测 `7.0e-5`；新的最佳 raw HF val_loss 是 `12.459524`。 |
| v0.3.7 | 2026-07-03 | 在 12-step 预算下验证 `5.0e-5`，并用 6-step 探测 `6.0e-5`；新的最佳 raw HF val_loss 是 `12.540532`。 |
| v0.3.6 | 2026-07-03 | 在 12-step 预算下验证 `4.0e-5`，并用 6-step 探测 `5.0e-5`；新的最佳 raw HF val_loss 是 `12.643408`。 |
| v0.3.5 | 2026-07-03 | 在 12-step 预算下比较 `3.0e-5`，并用 6-step 探测 `4.0e-5`；新的最佳 raw HF val_loss 是 `12.725430`。 |
| v0.3.4 | 2026-07-03 | 新增 `3.0e-5` 6-step 学习率探测和 `2.0e-5` 12-step 验证；当前最佳 raw HF val_loss 是 `13.097093`。 |
| v0.3.3 | 2026-07-03 | 新增中文 README，并把 MindSpeed-LLM 学习率边界搜索扩展到 `2.0e-5`；新的最佳 raw HF val_loss 是 `13.792945`。 |
| v0.3.2 | 2026-07-03 | 把 MindSpeed-LLM 学习率搜索扩展到 `7.5e-6` 和 `1.0e-5`；最佳 raw HF val_loss 更新为 `14.503275`。 |
| v0.3.1 | 2026-07-03 | 加入第一轮 MindSpeed-LLM 学习率搜索候选，并记录 `5.0e-6` 为当时最佳 6-step 结果。 |
| v0.3.0 | 2026-07-03 | 加入并验证 MindSpeed-LLM autoresearch 候选 runner，实现训练、转换、HF 评估、记录闭环。 |
| v0.2.2 | 2026-07-02 | 加入 MindSpeed MCore 到 HF 的转换，并用同一 HF 验证面评估转换后的 checkpoint。 |
| v0.2.1 | 2026-07-02 | 为 MindSpeed-LLM 增加验证集切分和评估参数，并记录第一次 validation-loss smoke。 |
| v0.2.0 | 2026-07-02 | 增加 MindSpeed-LLM adapter 和 Qwen3-0.6B 在昇腾 910C 上的框架评估记录。 |
| v0.1.0 | 2026-07-02 | 初始昇腾 Qwen3-0.6B autoresearch 原型、baseline 和第一次梯度累积搜索。 |

更多细节见 [CHANGELOG.md](CHANGELOG.md) 和 [docs/updates](docs/updates)。

## 更新规则

每次有意义的代码更新，都需要：

1. 更新 `CHANGELOG.md`；
2. 在 `docs/updates/` 下新增版本说明；
3. 保持英文和中文 README 的版本表同步；
4. 不提交 secrets、SSH 配置、模型文件、外部参考仓库、日志或运行缓存。

## 容器内快速启动

```bash
cd /workspace/ascend_autoresearch
bash run_train.sh
```

smoke test：

```bash
bash run_train.sh --time-budget 60
```

## 当前 HF Thin Loop 最佳结果

第一轮搜索只改变 `GRAD_ACCUM_STEPS`。

| Commit | val_loss | 显存 | 状态 | 改动 |
| --- | ---: | ---: | --- | --- |
| `724174c` | 6.411216 | 4.6 GB | keep | baseline |
| `50ecb19` | 6.305202 | 4.6 GB | keep | grad accumulation 8 -> 4 |
| `b523bcf` | 6.278897 | 4.6 GB | keep | grad accumulation 4 -> 2 |
| `b49232a` | 6.127654 | 4.6 GB | keep | grad accumulation 2 -> 1 |

当前 HF thin-loop 最佳是 `b49232a`，在 5 分钟探索预算下，相比 baseline 的
validation loss 约改善 4.42%。

## 框架评估快照

| 框架 | 能否跑通 | 效率 | 效果 |
| --- | --- | --- | --- |
| HF + torch_npu thin loop | 能 | 5 分钟预算可完成；观测到单 NPU 约 4.6 GB HBM。 | 最佳 val_loss：`6.127654`。 |
| MindSpeed-LLM | 能，已经跑通 train -> convert -> HF eval -> TSV record。 | Deepscaler smoke 热身后单步约 0.18-0.25 s；分配 HBM 约 10.3 GB。 | 当前最佳 raw HF val_loss 为 `12.420206`，对应 `LR=9.0e-5`、6 steps；base Qwen3 raw HF val_loss 为 `14.977717`。 |
| MindSpeed-MM | 暂未作为 Qwen3-0.6B 纯文本路径首选。 | 未测。 | 未测。 |

持续评估记录见 [docs/framework_evaluation.md](docs/framework_evaluation.md)。

## MindSpeed Autoresearch Loop

MindSpeed runner 执行一个完整候选闭环：

```bash
CANDIDATE_ENV=/workspace/framework_adapters/mindspeed_llm/candidates/baseline_6step.env \
  bash /workspace/framework_adapters/mindspeed_llm/run_autoresearch_candidate.sh
```

它会训练 MindSpeed 候选、把 checkpoint 转成 Hugging Face 格式、用固定 raw HF
验证脚本评估，并把结果追加到 `/workspace/runs/mindspeed_llm/results.tsv`。

当前 6-step MindSpeed 候选结果：

| Run | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: |
| `mindspeed_qwen3_0p6_baseline_6step` | 14.962966 | 0.612093 | 0.844955 |
| `mindspeed_qwen3_0p6_lr_low_6step` | 14.975369 | 0.615938 | 0.847379 |
| `mindspeed_qwen3_0p6_lr_high_6step` | 14.927566 | 0.590943 | 0.826132 |
| `mindspeed_qwen3_0p6_lr_higher_6step` | 14.819130 | 0.532507 | 0.772222 |
| `mindspeed_qwen3_0p6_lr_7p5em6_6step` | 14.622040 | 0.448289 | 0.693946 |
| `mindspeed_qwen3_0p6_lr_1em5_6step` | 14.503275 | 0.408763 | 0.643258 |
| `mindspeed_qwen3_0p6_lr_1p5em5_6step` | 14.061219 | 0.329362 | 0.547393 |
| `mindspeed_qwen3_0p6_lr_2em5_6step` | 13.792945 | 0.310141 | 0.513032 |
| `mindspeed_qwen3_0p6_lr_3em5_6step` | 13.120849 | 0.305617 | 0.491179 |
| `mindspeed_qwen3_0p6_lr_2em5_12step` | 13.097093 | 0.293768 | 0.477166 |
| `mindspeed_qwen3_0p6_lr_3em5_12step` | 12.725430 | 0.307299 | 0.481535 |
| `mindspeed_qwen3_0p6_lr_4em5_6step` | 12.786428 | 0.315171 | 0.509350 |
| `mindspeed_qwen3_0p6_lr_4em5_12step` | 12.668455 | 0.324619 | 0.492905 |
| `mindspeed_qwen3_0p6_lr_5em5_6step` | 12.643408 | 0.326887 | 0.528050 |
| `mindspeed_qwen3_0p6_lr_5em5_12step` | 12.686586 | 0.349943 | 0.516511 |
| `mindspeed_qwen3_0p6_lr_6em5_6step` | 12.540532 | 0.347458 | 0.555464 |
| `mindspeed_qwen3_0p6_lr_4p5em5_12step` | 12.678998 | 0.334108 | 0.503089 |
| `mindspeed_qwen3_0p6_lr_7em5_6step` | 12.459524 | 0.365993 | 0.577371 |
| `mindspeed_qwen3_0p6_lr_8em5_6step` | 12.434883 | 0.372704 | 0.590512 |
| `mindspeed_qwen3_0p6_lr_9em5_6step` | 12.420206 | 0.381243 | 0.601568 |

当前观测到的 MindSpeed 最佳：`lr_9em5_6step.env`。在同一个固定 HF 验证脚本下，
它相比 runner baseline 的 raw HF validation loss 改善 `2.542760`，相比 base
Qwen3-0.6B 改善 `2.557511`。
