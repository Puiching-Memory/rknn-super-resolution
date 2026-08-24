# Rockchip RKNN MLVC + Phase-RLFN 视频超分辨率

![SR 效果预览：Bicubic / Phase-RLFN SR / HR 参考及误差热图对比](docs/assets/sr_preview_grid.webp)

已在 RK3576、RK3588 和 RV1126B 上测试的 3× 视频超分辨率训练与部署工程。训练数据来自冻结
MLVC-S 的真实 P-frame 量化重建，而不是普通 bicubic 或传统视频解码帧。

```text
OpenVidHD mp4
  -> TorchCodec NVDEC + 360x640 canvas
  -> frozen MLVC-S (DPB + round quantization)
  -> current YUV444 frame + optional decoder ref_feature
  -> RGA bicubic base + BN-free Phase-RLFN residual
  -> 1080x1920
```

模型首先保证单帧 SR：当前帧始终独立经过 PixelUnshuffle(2)、局部残差特征网络、
PixelShuffle(6)，并在输出端叠加固定 bicubic 基线。MLVC 的 `ref_feature` 只通过
零初始化、无偏置的 codec adapter 提供增量信息；训练时随机丢弃 25% codec
特征，因此不提供额外信息时仍有完整的基础 SR 能力。

核心约束：

- 输入与目标默认是 MLVC BT.709 full-range YUV444 `[0,255]`。
- MLVC 内部把 360 高度补齐到 368，重建帧裁回 360；decoder feature 保留
  `96x46x80` 原始布局。
- SR 主干无 BatchNorm，训练图就是部署图，避免 DDP running stats 和融合误差。
- RGA bicubic、PixelUnshuffle/PixelShuffle、残差相加和 clip 位于 NPU 图外；
  RKNN core 输入为 `12x180x320` phase tensor，可选第二输入为
  `96x46x80` codec feature，输出为 `108x180x320` 有符号残差。
- 训练损失暂时保持 L1；所有阶段用同一 `vmaf`/`psnr` 验证协议选 best。

## 环境

项目只使用 `uv` 管理主环境，要求 Python 3.13，训练栈绑定 CUDA 13.2：

```bash
git clone --recurse-submodules https://github.com/Puiching-Memory/rknn-super-resolution.git
uv sync
uv run pytest
uv run ruff check src tests
```

`uv sync` 会通过构建钩子将 Netflix libvmaf 安装到 `.local/`。RKNN Toolkit 与
主环境隔离在 `.venv-rknn`，不要合并依赖。

准备 MLVC 权重和索引：

```bash
./scripts/setup_mlvc.sh
```

OpenVidHD 源视频放在 `data/OpenVidHD/` 任意子目录，文件名须与
`data/OpenVidHD/openvidhd_60k64_frame_sequences.csv` 对应。

## 配置与训练

默认配置是
`src/rknn_super_resolution/config/phase_rlfn_sr_x3.yaml`。主要参数：

```yaml
model:
  num_channels: 32
  num_blocks: 4
  phase_factor: 2
  codec_feature_channels: 96
  codec_project_channels: 16
data:
  sequence_frames: 8
  q_indices: [0, 21, 42, 63]
  codec_context: true
  codec_dropout: 0.25
training:
  float_lr: 1.0e-3
  qat_lr: 1.0e-5
  val_metric: vmaf
```

两卡训练示例：

```bash
./scripts/run_train.sh \
  --devices 6,7 \
  --nproc 2 \
  --save-dir checkpoints/phase-rlfn-codec-v1 \
  --experiment phase-rlfn-codec-v1
```

训练是一条 step 时间线：

```text
float -> qat_observe -> qat_stable -> complete
```

各阶段由验证平台期驱动，并保存真实最高分 checkpoint；`min_delta` 只控制平台期
patience，不会阻止小幅提升覆盖 best。恢复训练可设置：

```bash
RESUME=checkpoints/phase-rlfn-codec-v1/last.pth ./scripts/run_train.sh --devices 6,7
```

## ONNX 与 RKNN

导出 codec-aware 双输入 core：

```bash
uv run rknn-super-resolution export-onnx \
  --weight checkpoints/phase-rlfn-codec-v1/best_ema.pth \
  --from-qat --static --output phase_rlfn_sr_x3.onnx
```

传 `--no-codec-context` 可导出单输入 SR fallback core。生成双输入 PTQ 数据：

```bash
uv run rknn-super-resolution-build-rknn-calibration --samples 100
```

转换 RKNN：

```bash
uv run rknn-super-resolution convert-rknn \
  --onnx phase_rlfn_sr_x3.onnx \
  --output phase_rlfn_sr_x3.rknn \
  --target rk3576 \
  --input_size '12,180,320;96,46,80' \
  --calib_dir data/rknn_calib.txt
```

`--target` 会传给 RKNN Toolkit 的 `target_platform`（默认值由配置中的
`deploy.target` 决定），项目不对 Toolkit 支持的 target 设置白名单。已测试
`rk3576`、`rk3588` 和 `rv1126b`；RV1126B 需要 RKNN Toolkit2 2.3.2 或更高版本。
同一个 ONNX 应针对每种目标板卡分别编译，并使用不同的 `--output` 保存产物，例如
`phase_rlfn_sr_x3_rk3576.rknn`、`phase_rlfn_sr_x3_rk3588.rknn` 与
`phase_rlfn_sr_x3_rv1126b.rknn`。RKNN 专用解释器可通过
环境变量 `RKNN_PYTHON` 覆盖。

## 分布式不变量

每个 rank 独立执行 GPU 解码、MLVC 重建和 SR 前后向。验证先在所有 rank 上经过
DDP forward 并 gather；SwanLab、checkpoint 和预览只在 collective 完成后的
`rank0_section` 中执行，最终 early-stop 决策再 broadcast。

## 许可证

MIT License
