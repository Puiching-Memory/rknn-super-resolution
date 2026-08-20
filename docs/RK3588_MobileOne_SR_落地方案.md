# RK3576 Phase-MobileOne E 方案

本项目只维护一套模型与部署契约：E 方案。文件名保留是为了避免外部链接失效，
本文内容以 RK3576 实机 profile 结果为准。

## 模型结构

训练和评估接口保持常规 3× 超分：

```text
3×360×640 YCbCr444
  -> PixelUnshuffle(2)
12×180×320
  -> 3×3 stem, 12 -> 32
  -> MobileOne deploy block ×6, 32 channels
  -> global feature skip
  -> 3×3 output head, 32 -> 108
  -> Hardtanh[0,255]
108×180×320
  -> PixelShuffle(6)
3×1080×1920 YCbCr444
```

训练时前后重排位于 `MobileOneSR.forward()`，因此数据集、loss 和验证代码仍处理
普通 LR/HR 张量。导出只调用 `MobileOneSR.forward_core()`，ONNX/RKNN 图不包含
PixelUnshuffle 或 PixelShuffle。

## RK3576 部署契约

| 项目 | 固定值 |
| --- | --- |
| RKNN target | `rk3576` |
| RKNN input | NCHW `1×12×180×320` |
| RKNN output | NCHW `1×108×180×320` |
| CPU 前处理 | PixelUnshuffle(2) 通道重排 |
| CPU 后处理 | PixelShuffle(6) 通道重排 |
| NPU core | stem + 6 blocks + global skip + 3×3 head + clip |

CPU 重排必须使用 PyTorch PixelShuffle/PixelUnshuffle 的通道顺序。Python 参考实现
位于 `deploy/rknn_eval.py`，校准生成器输出同一布局的 NCHW `.npy`。

## 实机结果

RK3576 INT8 实测结果：

| 指标 | E 方案 |
| --- | ---: |
| 参数量（deploy graph） | 90,220 |
| MAC | 5.176 GMAC |
| NPU `run` | 5.07 ms |
| 端到端估算 | 12.63 ms |
| 吞吐 | 79.2 FPS |
| RKNN internal tensor memory | 13.824 MB |

端到端数字包含输入同步、输出同步以及 CPU 两次重排。它来自当前设备的 profile，
不是训练精度承诺；新模型仍需完成统一的 FP32→QAT 训练和验证集评估。阶段转换
由验证平台期驱动，不依赖固定训练步数。

## 环境边界

- GPU 训练机使用项目主 `uv` 环境。
- RK3576 板端不执行 `uv sync`，不安装 CUDA/PyTorch。
- ONNX 导出默认使用 CPU，也可在训练机显式指定 `--device cuda`。
- RKNN 转换使用隔离的 RKNN Toolkit 环境。

旧的 8-block、3 通道 RKNN 输入以及图内 PixelShuffle 路径均不再受支持。
