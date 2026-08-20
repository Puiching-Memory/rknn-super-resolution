# RK3576 MLVC + Phase-MobileOne 视频超分辨率

面向 RK3576 的 3× 视频超分辨率训练与部署工程。训练输入不是传统
H.264/H.265/AV1 解码帧，而是冻结 **MLVC-S** 在真实 P-frame 状态传播和
latent 量化之后的重建帧：

```text
OpenVidHD 连续帧
  -> 1920x1080 GT / Lanczos 下采样 640x360
  -> 冻结 MLVC-S（DPB + round 量化闭环，不执行 rANS）
  -> MobileOneSR x3
  -> 1920x1080
```

MLVC-S 要求空间尺寸对齐 16。对 640x360 输入，runtime 只在 MLVC 内部将
底部 replicate-pad 到 640x368，重建后裁回 640x360。补边不进入
MobileOneSR，也不改变 1920x1080 输出契约。

## 关键设计

- 数据集使用与 MLVC 相同的 OpenVidHD 60k x 64 frame sequences。
- 每个样本读取连续帧，首帧作为干净 DPB reference，后续帧保持
  `ref_frame`、`ref_feature` 闭环传播。
- MLVC 使用 eval 模式的真实 `round` 量化；训练直接调用 `compress_core()`，
  跳过 bit-estimate、rANS 和码流封装，因而不产生传统 codec cache。
- 默认从 MLVC 的 64 个质量点中采样 `[0, 21, 42, 63]`。
- 默认模型为 RK3576 实测筛选出的 E 方案：32 通道、6 个 MobileOne block、
  3×3 输出头。
- 训练接口保持 `3×360×640 -> 3×1080×1920`；部署时 CPU 执行
  `PixelUnshuffle(2)` / `PixelShuffle(6)`，NPU 图只处理
  `12×180×320 -> 108×180×320`，避免 PixelShuffle 在 NPU 上编译成慢速
  ConvTranspose。
- 模型默认在 MLVC BT.709 full-range YCbCr444 张量上训练。
- 训练保持 step-based DDP；验证、日志和 checkpoint 只在 collective 完成后
  进入 rank 0 段。
- 推理前执行 `switch_to_deploy()`，导出固定 360x640 输入的 ONNX/RKNN。

## 目录

```text
src/rk3588_mobile_sr/
├── config/                 默认 YAML 配置
├── data/
│   ├── openvid.py          OpenVidHD 连续序列读取
│   ├── mlvc_runtime.py     冻结 MLVC-S 无码流量化闭环
│   ├── mlvc_loader.py      CPU 读帧 + GPU MLVC batch processor
│   └── yuv_utils.py        BT.709 full-range YCbCr444
├── models/                 MobileOneSR 与 QAT
├── train/                  单入口、平台期驱动的 FP32→QAT 训练
├── distributed/            DDP 与 rank 0 同步原语
├── deploy/                 ONNX、RKNN 与精度验证
└── utils/                  指标、SwanLab、TraceML

scripts/
├── setup_mlvc.sh           固定 MLVC 源码、权重与 OpenVidHD 索引
├── run_train_8gpu.sh       统一 DDP 训练
└── setup_vmaf.sh           可选 VMAF 环境
```

项目已删除传统 codec manifest、离线 mp4/.npy cache、Snakemake 和
`--decode` 等旧接口。

## 环境

项目只使用 `uv` 管理主训练环境：

```bash
uv sync
uv run pytest
uv run ruff check src tests
```

Python 要求 3.12。PyTorch CUDA wheel 源以 `pyproject.toml` 为准。该环境只
用于 GPU 训练机；RK3576 板端不要执行 `uv sync`，无需安装 CUDA/PyTorch。
RKNN Toolkit 与主环境的 torch 版本冲突，应继续使用独立
`.venv-rknn`。

MLVC 不是主项目依赖包。本项目动态加载
`third_party/mlvc/video/src`，避免把 MLVC 自己的 torch 锁定并入主环境。

## 准备 MLVC

```bash
./scripts/setup_mlvc.sh
```

脚本会：

1. 将 MLVC 固定到 commit
   `e9f0114d71e886d7952af2a7a3c20b680443925f`。
2. 下载 `mlvc-s-psnr-v1.ckpt` 并校验官方 SHA256。
3. 下载官方 `openvidhd_60k64_frame_sequences.csv`。

源码和权重默认位于：

```text
third_party/mlvc/
data/mlvc/mlvc-s-psnr-v1.ckpt
```

## 准备 OpenVidHD

OpenVidHD 原视频体量很大，`setup_mlvc.sh` 不会自动拉取全量数据。
按照 MLVC 的
`video/BUILD_TRAIN_DATASET.md` 准备 parts 10-28，跳过
`*_part_ab`，并使用已下载的 60k x 64 CSV 执行官方步骤 6-8：

1. 从 Hugging Face 的 `nkp37/OpenVid-1M/OpenVidHD` 下载 parts 10-28。
2. 使用 MLVC `video/extract_frame_sequences.py` 抽取 WebP 连续帧。
3. 使用 `video/build_dataset_description.py` 生成 `description.json`。

最终目录必须满足：

```text
data/OpenVidHD/openvidhd_60k_train64/
├── description.json
├── frame_sequences.csv
├── <sequence_id_00000>/
│   ├── im00000.webp
│   ├── im00001.webp
│   └── ...
└── ...
```

`description.json` 中的 `path` 相对该文件所在目录解析。训练/验证按
sequence source 做稳定互斥切分，不会把同一 sequence 的帧分到两侧。

## 配置

默认配置位于
`src/rk3588_mobile_sr/config/mobileone_sr_x3.yaml`：

```yaml
model:
  scale: 3
  num_channels: 32
  num_blocks: 6
  phase_factor: 2
  output_kernel_size: 3

training:
  float_patience: 10
  float_min_delta: 0.1
  float_min_evaluations: 12
  observer_patience: 3
  observer_min_delta: 0.02
  observer_min_evaluations: 5
  qat_patience: 5
  qat_min_delta: 0.02
  qat_min_evaluations: 7

data:
  dataset_description: data/OpenVidHD/openvidhd_60k_train64/description.json
  mlvc_repo: third_party/mlvc
  mlvc_checkpoint: data/mlvc/mlvc-s-psnr-v1.ckpt
  mlvc_variant: small
  sequence_frames: 8
  q_indices: [0, 21, 42, 63]
  lr_size: [360, 640]
  hr_size: [1080, 1920]
  colorspace: yuv
  num_workers: 4
  mlvc_amp: true
```

CLI 可覆盖 `--dataset_description`、`--mlvc_repo`、
`--mlvc_checkpoint`、`--sequence_frames`、`--q_indices` 和
`--num_workers`。

## 统一训练

推荐使用八卡入口：

```bash
./scripts/run_train_8gpu.sh
```

或直接启动模块：

```bash
uv run torchrun --nproc_per_node=8 -m rk3588_mobile_sr.train.unified \
  --dataset_description data/OpenVidHD/openvidhd_60k_train64/description.json \
  --mlvc_repo third_party/mlvc \
  --mlvc_checkpoint data/mlvc/mlvc-s-psnr-v1.ckpt \
  --sequence_frames 8 \
  --q_indices 0 21 42 63 \
  --save_dir checkpoints/train
```

每个 GPU 都持有独立冻结 MLVC-S。CPU workers 只读取连续图像并完成一致
crop/flip，RGB->YCbCr、MLVC 重建和 patch crop 在对应 GPU 上执行。

训练是一个作业、一条全局 step 时间线和一个 checkpoint 根目录。状态机为：

```text
float --验证指标平台期--> qat_observe --平台期--> qat_stable --平台期--> complete
```

`float` 使用 VMAF（可用 `--no_vmaf` 改用 PSNR）；QAT 阶段使用 PSNR。
每次转换都有独立的 `patience`、`min_delta` 和 `min_evaluations`。配置中的
`*_safety_max_steps` 仅用于防止验证或数据异常造成无限运行，不是正常的阶段结束
条件。学习率也由同一验证指标通过 ReduceLROnPlateau 自动衰减。转换到 QAT 时
自动完成 BN 重校准、跨 rank BN 统计同步、部署融合和优化器重建；不再需要
`stage1_weight`。

最终用于导出的文件为 `checkpoints/train/best_ema.pth`。恢复训练只需传入任意
统一 checkpoint，程序会根据其中的 `phase` 自动重建 FP32 或 QAT 图：

```bash
RESUME=checkpoints/train/last.pth ./scripts/run_train_8gpu.sh
```

## ONNX 与 RKNN

导出前会融合 MobileOne 分支。E 方案导出的 ONNX 是纯 NPU phase core，而
不是包含 PixelShuffle 的完整训练图：

```bash
uv run rk3588-mobile-sr export-onnx \
  --weight checkpoints/train/best_ema.pth \
  --from-qat \
  --output mobileone_sr_x3.onnx \
  --static \
  --input_h 360 \
  --input_w 640
```

RKNN 模型输入和输出都是 phase-packed BT.709 full-range YCbCr444，而不是
RGB/NV12。PTQ 校准数据必须来自同一输入域。请在 GPU 训练机使用冻结 MLVC-S
生成 12 通道 `.npy` 校准样本：

```bash
uv run rk3588-build-rknn-calibration --samples 100
```

随后在独立 RKNN 环境转换：

```bash
uv run rk3588-mobile-sr convert-rknn \
  --onnx mobileone_sr_x3.onnx \
  --output mobileone_sr_x3.rknn \
  --target rk3576 \
  --input_size 12,180,320 \
  --calib_dir data/rknn_calib.txt \
  --weight checkpoints/train/best_ema.pth
```

板端先将 MLVC 重建的 `3×360×640` YCbCr444 张量按 PyTorch
PixelUnshuffle(2) 的通道顺序重排成 `12×180×320`，交给 RKNN；再将 RKNN 的
`108×180×320` 输出按 PixelShuffle(6) 重排为 `3×1080×1920`。这两个操作只是
内存重排，不做卷积。若 MLVC 内部使用 640x368 对齐面，必须先裁回有效 360
行。项目不保留旧的 3 通道 RKNN 图或包含 PixelShuffle 的部署路径。

## 训练同步不变量

```text
每个 step:
  所有 rank: MLVC reconstruction -> SR forward/backward -> optimizer
  log step: all_reduce loss -> rank 0 日志

每次验证:
  所有 rank: validate + gather
  rank0_section: SwanLab / preview / checkpoint
  所有 rank: broadcast early-stop decision
```

任何 rank 0 独占操作都必须放在 collective 完成之后并由
`rank0_section` 收尾。

## 许可证

MIT License
