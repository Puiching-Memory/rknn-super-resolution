# RK3588 MobileOne 超分辨率 (MobileOneSR)

面向 RK3588 NPU 部署的轻量级 3× 单帧超分辨率（SISR）方案：输入 360p（360×640），输出 1080p（1080×1920）。基于 MobileOne 重参数化卷积构建，支持 FP32 训练、知识蒸馏、QAT 量化感知训练，最终导出 ONNX 并转换为 RKNN INT8 模型在 RK3588 上推理。

## 功能特性

- **MobileOne 轻量骨干**：重参数化多分支结构，训练时多分支、推理时融合为单分支，兼顾精度与速度。
- **三阶段训练流程**：
  1. Stage 1：FP32 基线训练（L1 损失）
  2. Stage 2：知识蒸馏 + DCT 感知损失微调
  3. Stage 3：量化感知训练（QAT），产出 INT8 友好权重
- **端到端部署**：支持 PyTorch → ONNX → RKNN 转换链路。
- **RK3588 适配**：输入固定为 360×640，输出 1080×1920，便于接入视频后处理链路。
- **SwanLab 实验追踪**：使用 [SwanLab](https://github.com/SwanHubX/SwanLab) 记录 loss、PSNR 等训练指标，支持云端与本地可视化。
- **TraceML 性能诊断**：使用 [TraceML](https://github.com/traceopt-ai/traceml) 跟踪每步 input/compute 耗时、DDP rank 偏斜与显存趋势，训练结束输出 `final_summary.json`。

## 模型架构

```text
Input(3×360×640)
    ↓
Stem (Conv + BN + ReLU)
    ↓
MobileOne Block × N
    ↓
Global Skip Connection
    ↓
Out Conv → PixelShuffle(×3) → Hardtanh(0,255)
Output(3×1080×1920)
```

- 默认配置：`num_channels=32`, `num_blocks=8`, `num_conv_branches=4`, `scale=3`
- 推理前调用 `model.switch_to_deploy()` 融合 MobileOne 分支。

## 目录结构

```
rk3588_mobile_sr/
├── src/rk3588_mobile_sr/          # 可安装包（src layout）
│   ├── cli.py                     # 统一 CLI 入口
│   ├── config/                    # YAML 配置加载
│   ├── data/                      # DIV2K 数据加载（DALI / LMDB / PyTorch）
│   ├── models/                    # MobileOneSR 模型与 QAT 工具
│   ├── losses/                    # 训练损失函数
│   ├── utils/                     # 训练框架、指标、日志
│   ├── distributed/               # DDP 上下文、验证、同步原语
│   ├── train/                     # StepTrainer、TrainSession、Stage 1/2/3
│   ├── eval/                      # PSNR/SSIM 评测
│   └── deploy/                    # ONNX 导出 + RKNN 转换与精度评测
├── scripts/                       # 开发与性能分析脚本
├── tests/                         # 单元测试
├── docs/
├── pyproject.toml
└── README.md
```

## 环境安装

本项目使用 `uv` 管理依赖，Python 版本要求 `3.12.*`，训练环境固定为 **CUDA 13.0（cu130）**，需要 NVIDIA GPU。

```bash
uv sync
```

开发依赖（ruff、pytest、pre-commit）：

```bash
uv sync
```

PyTorch 与 DALI 的 wheel 源已在 `pyproject.toml` 中配置（`cu130` + NVIDIA PyPI）。
> RKNN 转换建议在独立环境中安装 `rknn-toolkit2`，因为它对 torch/onnx 的版本限制与本项目训练依赖冲突。

## CLI 用法

安装后可通过统一 CLI 调用各子命令（参数与原先脚本一致）：

```bash
# 查看帮助
uv run rk3588-mobile-sr --help

# 训练（DDP 仍用 torchrun 包装）
torchrun --nproc_per_node=8 rk3588-mobile-sr train stage1 \
  --hr_dir data/DIV2K_train_HR \
  --lr_dir data/DIV2K_train_LR_bicubic/X3 \
  ...

# 评测 / 导出
uv run rk3588-mobile-sr eval --weight checkpoints/stage2/best.pth ...
uv run rk3588-mobile-sr export-onnx --weight checkpoints/stage2/best.pth ...
```

也保留了独立入口别名：`rk3588-train-stage1`、`rk3588-eval-psnr` 等。

## 数据准备

使用 DIV2K 数据集，目录结构如下：

```
data/
├── DIV2K_train_HR/
├── DIV2K_train_LR_bicubic/X3/
├── DIV2K_valid_HR/
└── DIV2K_valid_LR_bicubic/X3/
```

## 训练流程

训练脚本使用 [SwanLab](https://github.com/SwanHubX/SwanLab) 记录实验指标。DDP 训练仅在 rank 0 上初始化 SwanLab，日志写入 `<save_dir>/swanlog/` 并同步至 SwanLab 云端（首次使用需按提示登录）。

训练过程文本日志由 [loguru](https://github.com/Delgan/loguru) 写入 `<save_dir>/train.log`（每次启动覆盖），rank 0 在交互式终端下会同步输出到 stderr。**无需** `tee` 或 `TRAIN_PLAIN_LOG`；直接 `torchrun` 启动即可。

### DDP 同步模型（step-based）

三阶段训练共用同一套 **step-based** 循环（`StepTrainer`），由 `DistributedContext` 封装所有 collective，避免 rank 0 在验证后做日志/SwanLab/checkpoint 时与其他 rank 的 `broadcast` 死锁。

```text
每个 training step:
  所有 rank: forward + backward + optimizer.step
  每 log_every: all_reduce_avg(loss) → rank0 写日志

每 val_every step:
  所有 rank: validate_ddp(_extended) + gather 指标
  rank0_section:
    rank0: deploy 检查 / SwanLab 图像 / best.pth
    所有 rank: barrier
  所有 rank: broadcast_bool(should_stop)  # 早停决策
```

核心原语 `rank0_section`：**任何 rank0 独占逻辑必须在 collective 完成之后、且由 barrier 收尾**，禁止在 `broadcast` 之前让非 0 rank 等待。

Stage 1 / Stage 2 默认启用 **早停**（验证 PSNR 连续若干次无提升则停止）；Stage 3 不启用早停。各阶段 `--max_steps` 为安全上限。

```bash
# 禁用 SwanLab
torchrun --nproc_per_node=1 rk3588-mobile-sr train stage1 ... --no_swanlab

# 指定实验名
torchrun --nproc_per_node=1 rk3588-mobile-sr train stage1 ... --swanlab_experiment my-run-001
```

### TraceML 训练性能诊断

训练脚本已集成 [TraceML](https://github.com/traceopt-ai/traceml)。用 `traceml run` 启动（替代 `torchrun`/`python`），训练结束后会在 `logs/<run_name>/` 生成 `final_summary.json` 与可读文本报告，用于定位 DataLoader 瓶颈、DDP rank 偏斜、显存泄漏等问题。

```bash
# 单卡
traceml run rk3588-mobile-sr train stage1 \
  --hr_dir data/DIV2K_train_HR \
  --lr_dir data/DIV2K_train_LR_bicubic/X3 \
  --val_hr_dir data/DIV2K_valid_HR \
  --val_lr_dir data/DIV2K_valid_LR_bicubic/X3 \
  --save_dir ./checkpoints/stage1

# 多卡 DDP（TraceML 内置 --nproc-per-node，等价于 torchrun）
traceml run rk3588-mobile-sr train stage1 --nproc-per-node=8 \
  --hr_dir data/DIV2K_train_HR \
  --lr_dir data/DIV2K_train_LR_bicubic/X3 \
  --val_hr_dir data/DIV2K_valid_HR \
  --val_lr_dir data/DIV2K_valid_LR_bicubic/X3 \
  --save_dir ./checkpoints/stage1

# 实时终端面板
traceml run rk3588-mobile-sr train stage1 --mode=cli --nproc-per-node=8 ...

# 对比两次运行（例如调 num_workers 前后）
traceml compare logs/run_a/final_summary.json logs/run_b/final_summary.json
```

- 通过 `traceml run` 启动时 TraceML **默认开启**；普通 `torchrun` 下可用 `--traceml` 手动开启 instrumentation，但完整 summary 仍需 `traceml run`。
- 可用 `--no-traceml` 关闭；summary 指标会以 `traceml/...` 前缀同步到 SwanLab（若未禁用）。
- Stage 2 / Stage 3 同样支持：`traceml run rk3588-mobile-sr train stage2 ...`、`traceml run rk3588-mobile-sr train stage3-qat ...`。

### Stage 1：FP32 基线

```bash
torchrun --nproc_per_node=8 rk3588-mobile-sr train stage1 \
  --hr_dir data/DIV2K_train_HR \
  --lr_dir data/DIV2K_train_LR_bicubic/X3 \
  --val_hr_dir data/DIV2K_valid_HR \
  --val_lr_dir data/DIV2K_valid_LR_bicubic/X3 \
  --patch_size 128 \
  --batch_size 16 \
  --lr 1e-3 \
  --save_dir ./checkpoints/stage1
```

Stage 1 默认启用 **早停**：每 `--val_every`（1000）step 验证一次，若连续 `--early_stop_patience`（10）次验证 PSNR 无提升（阈值 `--early_stop_min_delta` 0.01 dB）则停止；`--max_steps`（100000）为安全上限。可用 `--no_early_stop` 改为只跑到 `max_steps`。

### Stage 2：蒸馏 + 感知损失微调

与 Stage 1 相同，采用 **step-based** 训练（LMDB 无限随机采样）：默认 `--max_steps 80000`，每 `--val_every 4000` step 验证，每 `--log_every 500` step 打日志。默认启用 **早停**：连续 `--early_stop_patience`（8）次验证 PSNR 无提升（阈值 `--early_stop_min_delta` 0.005 dB）则停止；可用 `--no_early_stop` 跑满 `max_steps`。需先准备教师模型权重（MambaIRv2Light ×3），默认路径 `checkpoints/teacher/mambairv2_lightSR_x3.pth`：

```bash
# GitHub 下载需代理时先设置（按你的代理地址修改）
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897

uv run python scripts/download_mambairv2_teacher.py
```

```bash
torchrun --nproc_per_node=1 rk3588-mobile-sr train stage2 \
  --hr_dir data/DIV2K_train_HR \
  --lr_dir data/DIV2K_train_LR_bicubic/X3 \
  --val_hr_dir data/DIV2K_valid_HR \
  --val_lr_dir data/DIV2K_valid_LR_bicubic/X3 \
  --teacher_arch mambairv2_light \
  --teacher_weight checkpoints/teacher/mambairv2_lightSR_x3.pth \
  --stage1_weight checkpoints/stage1/best.pth \
  --save_dir ./checkpoints/stage2
```

### Stage 3：QAT 量化感知训练

同样为 **step-based**。QAT 分三阶段，由 step 控制切换（非 epoch）：

| 阶段 | step 范围 | 行为 |
|------|-----------|------|
| Phase 1 | 1 – `phase1_steps` (3000) | 训练 + 更新 observer |
| Phase 2 | `phase1_steps+1` – `phase2_steps` (9000) | 冻结 observer，继续 fake-quant 训练 |
| Phase 3 | `phase2_steps+1` – `max_steps` (15000) | 冻结 fake-quant，权重微调 |

```bash
torchrun --nproc_per_node=1 rk3588-mobile-sr train stage3-qat \
  --stage2_weight checkpoints/stage2/best.pth \
  --hr_dir data/DIV2K_train_HR \
  --lr_dir data/DIV2K_train_LR_bicubic/X3 \
  --max_steps 15000 \
  --phase1_steps 3000 \
  --phase2_steps 9000 \
  --save_dir ./checkpoints/stage3_qat
```

## 导出与评测

### 导出 ONNX

```bash
# FP32 ONNX
uv run rk3588-mobile-sr export-onnx \
  --weight checkpoints/stage2/best.pth \
  --output mobileone_sr_x3.onnx \
  --input_h 360 \
  --input_w 640

# QAT ONNX
uv run rk3588-mobile-sr export-onnx \
  --weight checkpoints/stage3_qat/best.pth \
  --output mobileone_sr_x3_qat.onnx \
  --qat \
  --backend qnnpack \
  --input_h 360 \
  --input_w 640
```

### 评测 PSNR/SSIM

```bash
uv run rk3588-mobile-sr eval \
  --weight checkpoints/stage2/best.pth \
  --hr_dir data/DIV2K_valid_HR \
  --lr_dir data/DIV2K_valid_LR_bicubic/X3 \
  --save_dir results/stage2
```

## RKNN 部署转换

> RKNN 转换与 FP32/INT8 精度对比见 `src/rk3588_mobile_sr/deploy/rknn.py`（需在独立 rknn-toolkit2 环境中运行）。

```bash
uv run rk3588-mobile-sr convert-rknn \
  --onnx mobileone_sr_x3.onnx \
  --output mobileone_sr_x3.rknn \
  --target rk3588 \
  --calib_dir data/rknn_calib.txt
```

转换完成后，将 `mobileone_sr_x3.rknn` 部署到 RK3588 上通过 `librknnrt.so` 进行推理。

## 配置说明

`src/rk3588_mobile_sr/config/mobileone_sr_x3.yaml` 中集中管理模型、数据、训练和部署参数，可通过 Python API 加载：

```python
from rk3588_mobile_sr.config import load_config
cfg = load_config()  # 或 load_config("path/to/custom.yaml")
```

示例配置：

```yaml
model:
  scale: 3
  num_channels: 32
  num_blocks: 8
  num_conv_branches: 4

deploy:
  input_h: 360
  input_w: 640
  onnx_output: "mobileone_sr_x3.onnx"
  rknn_output: "mobileone_sr_x3.rknn"
```

## 性能参考

典型指标（以实际训练结果为准）：

| 阶段    | 输入分辨率 | 输出分辨率 | 量化 | 部署方式          |
| ------- | ---------- | ---------- | ---- | ----------------- |
| Stage 2 | 360×640    | 1080×1920  | FP32 | ONNX / PyTorch    |
| Stage 3 | 360×640    | 1080×1920  | INT8 | RKNN (RK3588 NPU) |

详细的落地方案与性能分析见 [docs/RK3588_MobileOne_SR_落地方案.md](docs/RK3588_MobileOne_SR_落地方案.md)。

## 许可证

MIT License
