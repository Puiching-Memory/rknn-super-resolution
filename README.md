# RK3588 MobileOne 超分辨率 (MobileOneSR)

面向 RK3588 NPU 部署的轻量级 3× 单帧超分辨率（SISR）方案：输入 360p（360×640），输出 1080p（1080×1920）。基于 MobileOne 重参数化卷积构建，支持 FP32 训练、知识蒸馏、QAT 量化感知训练，最终导出 ONNX 并转换为 RKNN INT8 模型在 RK3588 上推理。

## 功能特性

- **MobileOne 轻量骨干**：重参数化多分支结构，训练时多分支、推理时融合为单分支，兼顾精度与速度。
- **两阶段训练流程**：
  1. Stage 1：FP32 基线训练（L1 损失）
  2. Stage 2：量化感知训练（QAT），产出 INT8 友好权重
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
│   ├── data/                      # 训练期 manifest / loader（codec_index、DALI）
│   ├── data_pipeline/             # 离线 codec cache 构建（Snakemake 配套 Python）
│   ├── models/                    # MobileOneSR 模型与 QAT 工具
│   ├── losses/                    # 训练损失函数
│   ├── utils/                     # 训练框架、指标、日志
│   ├── distributed/               # DDP 上下文、验证、同步原语
│   ├── train/                     # StepTrainer、TrainSession、Stage 1/2
│   ├── eval/                      # PSNR/SSIM 评测
│   └── deploy/                    # ONNX 导出 + RKNN 转换与精度评测
├── scripts/                       # 运维 bash 入口 + pipeline/Snakefile
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

# 训练（DDP 仍用 torchrun 包装；canvas codec 离线缓存 + NVDEC）
torchrun --nproc_per_node=3 rk3588-mobile-sr train stage1 \
  --codec_manifest data/codec_cache/manifest.jsonl \
  --val_manifest data/sources/manifests/val_fixed.jsonl \
  --decode auto \
  ...

# 评测 / 导出
uv run rk3588-mobile-sr eval --weight checkpoints/stage1/best.pth ...
uv run rk3588-mobile-sr export-onnx --weight checkpoints/stage1/best.pth ...
```

也保留了独立入口别名：`rk3588-train-stage1`、`rk3588-eval-psnr` 等。

## 数据准备

训练数据由 **manifest JSONL** 描述原生视频源（UVG 1080p YUV420p）。离线 LR codec cache 由 **Snakemake** 编排（代码在 `src/rk3588_mobile_sr/data_pipeline/`，DAG 在 `scripts/pipeline/`），训练侧读 `data/codec_cache/manifest.jsonl`。

```bash
# 一键：发现源 -> 写 manifest -> Snakemake 构建 codec cache
./scripts/build_data.sh

# 或分步：
uv run rk3588-build-train-manifest    # train.jsonl + val_fixed.jsonl（同一 discover 脚本）
uv run rk3588-build-codec-cache       # snakemake -j$(nproc) all

# 仅刷新 manifest（已有 mp4，不重编码）：
uv run python -m rk3588_mobile_sr.data_pipeline.write_codec_manifest \
  --root . \
  --train-manifest data/sources/manifests/train.jsonl \
  --output data/codec_cache/manifest.jsonl \
  --cache-dir data/codec_cache \
  --hr-dir data/hr_lossless
```

`build_data.sh` 支持环境变量：`WORKERS`、`CLIPS_PER_VIDEO`、`BITRATES`；干跑传 `./scripts/build_data.sh -- -n`。

目录结构示例：

```
data/
├── sources/manifests/
│   ├── train.jsonl              # UVG YUV 源
│   └── val_fixed.jsonl          # 固定 UVG 验证行
├── codec_cache/manifest.jsonl   # 离线 LR clip 索引（训练读取）
├── codec_cache/*.mp4            # 预编码 LR clips（退化+压缩）
├── hr_lossless/*_s{clip}_hr.mp4 # 每 clip 的无损 H.264 HR（CRF 0，监督标签）
└── UVG_raw/yuv_1080p/           # 原始 1080p YUV420 序列

scripts/pipeline/
├── Snakefile                    # discover -> hr_clip -> degrade_and_encode -> manifest
├── config.yaml                  # codecs / bitrates / gop_candidates / clips 等
└── ffmpeg/                      # hr_clip.sh（无损 HR 提取）
```

默认分辨率契约（与 deploy 对齐）：HR **1920×1080**，LR **640×360** canvas。训练数据管线为 **离线 codec cache + GPU 解码**：

1. Snakemake 从原始 YUV 提取每 clip 的 **无损 H.264 HR**（CRF 0，监督标签与源 bit-exact），再对 HR 做 **混合下采样核 + 压缩前传感器噪声 + LR codec 编码**（单次 ffmpeg pass，无 rgb24 中间产物），写出 `codec_cache/manifest.jsonl`
2. 训练时 `decode=auto` 使用 DALI NVDEC（需 `libnvcuvid`；torchcodec CPU fallback 已移除）

退化链顺序模拟真实视频采集：HR 无损 -> 光学低通模糊 -> 传感器噪声 -> 混合下采样核（area/bicubic/lanczos）-> codec 编码；在线 augment 只保留压缩后的 JPEG/解码噪声，物理顺序正确。GOP 与码率按真实分布采样（log-normal 码率、加权 GOP 候选），I/P 帧在 `expand_codec_clip_frames` 中加权（P 帧权重更高以多见块效应）。

Docker 需设置 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` 以启用 NVDEC。

### Bash 脚本（`scripts/`）

推荐用 `scripts/` 下的 bash 入口启动训练与数据准备（内部统一 `uv run` + `torchrun -m ...`）：

| 脚本                                  | 用途                                            |
| ------------------------------------- | ----------------------------------------------- |
| `./scripts/build_data.sh`             | Snakemake 构建 train/val manifest + codec cache |
| `./scripts/run_stage1_8gpu.sh`        | Stage 1 八卡 DDP 训练                           |
| `./scripts/bench_dataloader.sh`       | 单卡 DataLoader 吞吐                            |
| `./scripts/bench_ddp.sh`              | 多卡 DDP step 吞吐                              |
| `./scripts/profile_stage1.sh`         | 数据 vs 计算耗时剖析                            |
| `./scripts/generate_report_charts.sh` | 从 `stage1_metrics.json` 生成报告图             |

环境变量示例：`SAVE_DIR`、`NPROC`、`RESUME`、`TRAIN_EXPERIMENT_NAME`、`EXTRA_ARGS`。

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
  --codec_manifest data/codec_cache/manifest.jsonl \
  --val_manifest data/sources/manifests/val_fixed.jsonl \
  --save_dir ./checkpoints/stage1

# 多卡 DDP（TraceML 内置 --nproc-per-node，等价于 torchrun）
traceml run rk3588-mobile-sr train stage1 --nproc-per-node=8 \
  --codec_manifest data/codec_cache/manifest.jsonl \
  --val_manifest data/sources/manifests/val_fixed.jsonl \
  --save_dir ./checkpoints/stage1

# 实时终端面板
traceml run rk3588-mobile-sr train stage1 --mode=cli --nproc-per-node=8 ...

# 对比两次运行（例如调 num_workers 前后）
traceml compare logs/run_a/final_summary.json logs/run_b/final_summary.json
```

- 通过 `traceml run` 启动时 TraceML **默认开启**；普通 `torchrun` 下可用 `--traceml` 手动开启 instrumentation，但完整 summary 仍需 `traceml run`。
- 可用 `--no-traceml` 关闭；summary 指标会以 `traceml/...` 前缀同步到 SwanLab（若未禁用）。
- Stage 2 同样支持：`traceml run rk3588-mobile-sr train stage2 ...`。

### Stage 1：FP32 基线

```bash
# 推荐：bash 入口（日志写入 checkpoints/stage1/console.log）
./scripts/run_stage1_8gpu.sh

# 或手动 torchrun（注意用 -m 模块路径，勿直接写 rk3588-mobile-sr 文件名）
uv run torchrun --nproc_per_node=8 -m rk3588_mobile_sr.train.stage1 \
  --codec_manifest data/codec_cache/manifest.jsonl \
  --val_manifest data/sources/manifests/val_fixed.jsonl \
  --decode auto \
  --batch_size 16 \
  --lr 1e-3 \
  --save_dir ./checkpoints/stage1
```

Stage 1 默认启用 **早停**：每 `--val_every`（1000）step 验证一次，若连续 `--early_stop_patience`（10）次验证 PSNR 无提升（阈值 `--early_stop_min_delta` 0.01 dB）则停止；`--max_steps`（100000）为安全上限。可用 `--no_early_stop` 改为只跑到 `max_steps`。

### Stage 2：QAT 量化感知训练

同样为 **step-based**。QAT 分三阶段，由 step 控制切换（非 epoch）：

| 阶段    | step 范围                                | 行为                                |
| ------- | ---------------------------------------- | ----------------------------------- |
| Phase 1 | 1 – `phase1_steps` (3000)                | 训练 + 更新 observer                |
| Phase 2 | `phase1_steps+1` – `phase2_steps` (9000) | 冻结 observer，继续 fake-quant 训练 |
| Phase 3 | `phase2_steps+1` – `max_steps` (15000)   | 冻结 fake-quant，权重微调           |

```bash
torchrun --nproc_per_node=1 rk3588-mobile-sr train stage2 \
  --stage1_weight checkpoints/stage1/best.pth \
  --codec_manifest data/codec_cache/manifest.jsonl \
  --max_steps 15000 \
  --phase1_steps 3000 \
  --phase2_steps 9000 \
  --save_dir ./checkpoints/stage2_qat
```

## 导出与评测

### 导出 ONNX

```bash
# FP32 ONNX
uv run rk3588-mobile-sr export-onnx \
  --weight checkpoints/stage1/best.pth \
  --output mobileone_sr_x3.onnx \
  --input_h 360 \
  --input_w 640

# QAT ONNX
uv run rk3588-mobile-sr export-onnx \
  --weight checkpoints/stage2_qat/best.pth \
  --output mobileone_sr_x3_qat.onnx \
  --qat \
  --backend qnnpack \
  --input_h 360 \
  --input_w 640
```

### 评测 PSNR/SSIM

```bash
uv run rk3588-mobile-sr eval \
  --weight checkpoints/stage1/best.pth \
  --hr_dir data/UVG_raw/yuv_1080p \
  --lr_dir data/codec_cache \
  --save_dir results/stage1
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
| Stage 1 | 360×640    | 1080×1920  | FP32 | ONNX / PyTorch    |
| Stage 2 | 360×640    | 1080×1920  | INT8 | RKNN (RK3588 NPU) |

详细的落地方案与性能分析见 [docs/RK3588_MobileOne_SR_落地方案.md](docs/RK3588_MobileOne_SR_落地方案.md)。

## 许可证

MIT License
