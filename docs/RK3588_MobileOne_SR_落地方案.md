# Rockchip RKNN RGA-Bicubic + Phase-Residual MobileOne 方案

本项目只维护 RGA-Bicubic + Phase-Residual MobileOne 这一套模型与部署契约。

## 模型结构

训练和评估接口保持常规 3× 超分：

```text
3×360×640 YCbCr444
  |-------------------------> bicubic x3 (RGA) -------------------|
  -> PixelUnshuffle(2)                                           |
12×180×320
  -> spatial stem, 12 -> 32 -------------------------------------+
2× historical Y
  -> PixelUnshuffle(2), 8 channels
  -> subtract current Y phases
  -> temporal adapter, 8 -> 32 ----------------------------------+
  -> feature add
  -> MobileOne deploy block ×6, 32 channels
  -> global feature skip
  -> zero-init 3×3 residual head, 32 -> 108
108×180×320
  -> PixelShuffle(6)
3×1080×1920 signed residual
  -> add RGA bicubic base -> clip[0,255]
3×1080×1920 YCbCr444
```

训练时前后重排位于 `MobileOneSR.forward()`，因此数据集、loss 和验证代码仍处理
普通 LR/HR 张量，损失仍为 L1。训练样本携带当前 YUV 和两帧历史 Y，并随机将
历史替换为当前 Y，使 temporal adapter 在训练期间可被严格旁路。PyTorch bicubic
是 RGA 路径的训练参考；输出头零初始化使新模型在 step 0 精确等于 bicubic。导出只调用
`MobileOneSR.forward_core()`，ONNX/RKNN 图不包含 bicubic、PixelUnshuffle、
PixelShuffle、add 或 clip。

## Rockchip RKNN 部署契约

| 项目               | 固定值                                                              |
| ------------------ | ------------------------------------------------------------------- |
| RKNN target        | 不设白名单；已测试 `rk3576`、`rk3588` 和 `rv1126b`，每种板卡单独编译 |
| RKNN VSR input     | NCHW `1×20×180×320`                                                 |
| RKNN SR-only input | NCHW `1×12×180×320`                                                 |
| RKNN output        | NCHW `1×108×180×320` signed residual                                |
| CPU 前处理         | PixelUnshuffle(2) 通道重排                                          |
| 并行主路径         | RGA bicubic，`360×640 -> 1080×1920`                                 |
| CPU 后处理         | PixelShuffle(6) + add + clip 融合 kernel                            |
| NPU core           | spatial stem + optional temporal adapter + 6 blocks + residual head |

CPU 重排必须使用 PyTorch PixelShuffle/PixelUnshuffle 的通道顺序。RGA 与 NPU
应从同一输入并行启动，等待二者完成后执行一次融合的 unpack/add/clip。Python
参考实现位于 `deploy/rknn_eval.py`，校准生成器输出同一布局的 NCHW `.npy`。

## 性能预算

NPU residual core 的卷积规模为：

| 指标                   |    SR-only |        VSR |
| ---------------------- | ---------: | ---------: |
| 参数量（deploy graph） |     90,220 |     92,524 |
| MAC                    | 5.176 GMAC | 5.309 GMAC |
| 额外历史帧缓存         |          0 |  2×Y plane |

端到端必须按本契约重新实测：RGA 与 NPU 并行，计入同步和融合
unpack/add/clip。模型仍需完成统一的 FP32→QAT 训练和验证集评估；阶段转换由
验证平台期驱动，不依赖固定训练步数。

## 环境边界

- GPU 训练机使用项目主 `uv` 环境。
- 目标板端不执行 `uv sync`，不安装 CUDA/PyTorch。
- ONNX 导出默认使用 CPU，也可在训练机显式指定 `--device cuda`。
- RKNN 转换使用隔离的 RKNN Toolkit 环境。

训练、导出与部署只接受本文定义的 signed phase-residual 契约。
