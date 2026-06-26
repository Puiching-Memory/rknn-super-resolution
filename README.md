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
├── configs/
│   └── mobileone_sr_x3.yaml       # 默认训练/部署配置
├── data/
│   └── div2k_loader.py            # DIV2K 数据集加载器
├── docs/
│   └── RK3588_MobileOne_SR_落地方案.md
├── losses/
│   ├── charbonnier.py
│   ├── dct_loss.py
│   └── kd_loss.py
├── models/
│   ├── mobileone_block.py         # MobileOne 重参数化块
│   ├── mobileone_sr.py            # MobileOneSR 模型定义
│   ├── qat_utils.py               # QAT 工具
│   └── teacher_wrapper.py         # 教师模型蒸馏包装
├── eval_psnr.py                   # PSNR/SSIM 评测
├── export_onnx.py                 # 导出 ONNX
├── rknn_convert.py                # ONNX → RKNN 模板脚本
├── train_stage1.py                # 阶段 1 训练
├── train_stage2.py                # 阶段 2 蒸馏微调
├── train_stage3_qat.py            # 阶段 3 量化感知训练
├── pyproject.toml
└── README.md
```

## 环境安装

本项目使用 `uv` 管理依赖，Python 版本要求 `3.12.*`（见 `pyproject.toml`）。

```bash
# CPU 版本（默认）
uv sync --extra cpu

# CUDA 版本
uv sync --extra cuda --extra-index-url https://download.pytorch.org/whl/cu130
```

> 注意：PyTorch 的 CPU/CUDA 源互斥，需在命令行显式指定 `--extra-index-url`。
> RKNN 转换建议在独立环境中安装 `rknn-toolkit2`，因为它对 torch/onnx 的版本限制与本项目训练依赖冲突。

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

### Stage 1：FP32 基线

```bash
python train_stage1.py \
  --hr_dir data/DIV2K_train_HR \
  --lr_dir data/DIV2K_train_LR_bicubic/X3 \
  --val_hr_dir data/DIV2K_valid_HR \
  --val_lr_dir data/DIV2K_valid_LR_bicubic/X3 \
  --patch_size 128 \
  --batch_size 16 \
  --epochs 600 \
  --lr 1e-3 \
  --save_dir ./checkpoints/stage1
```

### Stage 2：蒸馏 + 感知损失微调

需先准备教师模型权重（如 EDSR），默认路径 `checkpoints/teacher/edsr_x3.pth`。

```bash
python train_stage2.py \
  --hr_dir data/DIV2K_train_HR \
  --lr_dir data/DIV2K_train_LR_bicubic/X3 \
  --val_hr_dir data/DIV2K_valid_HR \
  --val_lr_dir data/DIV2K_valid_LR_bicubic/X3 \
  --teacher_weight checkpoints/teacher/edsr_x3.pth \
  --save_dir ./checkpoints/stage2
```

### Stage 3：QAT 量化感知训练

```bash
python train_stage3_qat.py \
  --weight checkpoints/stage2/best.pth \
  --hr_dir data/DIV2K_train_HR \
  --lr_dir data/DIV2K_train_LR_bicubic/X3 \
  --save_dir ./checkpoints/stage3_qat
```

## 导出与评测

### 导出 ONNX

```bash
# FP32 ONNX
python export_onnx.py \
  --weight checkpoints/stage2/best.pth \
  --output mobileone_sr_x3.onnx \
  --input_h 360 \
  --input_w 640

# QAT ONNX
python export_onnx.py \
  --weight checkpoints/stage3_qat/best.pth \
  --output mobileone_sr_x3_qat.onnx \
  --qat \
  --backend qnnpack \
  --input_h 360 \
  --input_w 640
```

### 评测 PSNR/SSIM

```bash
python eval_psnr.py \
  --weight checkpoints/stage2/best.pth \
  --hr_dir data/DIV2K_valid_HR \
  --lr_dir data/DIV2K_valid_LR_bicubic/X3 \
  --save_dir results/stage2
```

## RKNN 部署转换

> `rknn_convert.py` 当前为模板脚本，需要在已安装 `rknn-toolkit2` 的环境中补全 RKNN API 调用。

```bash
python rknn_convert.py \
  --onnx mobileone_sr_x3.onnx \
  --output mobileone_sr_x3.rknn \
  --target rk3588 \
  --calib_dir data/rknn_calib.txt
```

转换完成后，将 `mobileone_sr_x3.rknn` 部署到 RK3588 上通过 `librknnrt.so` 进行推理。

## 配置说明

`configs/mobileone_sr_x3.yaml` 中集中管理模型、数据、训练和部署参数，例如：

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
