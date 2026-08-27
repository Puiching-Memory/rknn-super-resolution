# Codec-aware VSR 研究归档：单帧局部改造 → entropy 门控 → 最终 `y_hat` 实验

- 归档日期：2026-08-27（UTC）
- 合并来源（原文移入 `docs/archive/`，以本文为唯一现行记录）：
  - `archive/20260825_failed_experiments_summary.md`
  - `archive/20260826_igr_phase_sr_research_plan.md`
  - `archive/20260827_codec_aware_vsr_final_experiment.md`
- 基线分支：`master`；仓库 `rknn-super-resolution`
- 本文性质：第 3、4 章为**历史事实**；第 5 章实验已启动但在完成前被手动中断，**验收门槛未执行，最终结论未定**
- 全程核心原则：先证明"正确历史有用"，再讨论联合训练、QAT 与部署

## 1. 一页结论

### 1.1 时间线

| 阶段     | 日期      | 内容                                                                                       | 结果                                                                                                              |
| -------- | --------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| 起点     | ~08-23 前 | 单帧 Phase-RLFN + MLVC feature 部署基线 `phase-rlfn-codec-v1`                              | QAT stable VMAF 70.67 / PSNR 34.93                                                                                |
| 第一阶段 | 08-24～25 | memory/q-router、辅助 loss、边缘重参数化、QAT KD 等 5 类假设的消融与对照                   | 除 KD 的量化保真技巧外全部失败；代码撤销，产物保留                                                                |
| 第二阶段 | 08-26     | IGR-PhaseSR：`y_raw + scale-index` entropy innovation 驱动显式 SR state（主干冻结）        | 完整完成 10,000 step；相对 matched control 仅 +0.0934 VMAF（< +0.10 停止线），倒序历史仅 -0.0036，机制门槛未通过  |
| 第三阶段 | 08-27     | 主救援实验：改用完整重建 latent `y_hat` 作为 innovation，同一 recurrent cell、同一验收协议 | 训练启动后于 step ≈5,500 被手动中断（SIGINT）；best 为 step 5,000（val VMAF≈70.57）。验收消融未执行，最终判伪悬置 |

### 1.2 已排除的路线

1. 在单帧 Phase-RLFN 上继续叠加 MLVC feature adapter、q-router、辅助 loss、边缘分支或 QAT 技巧，不能建立真正的视频超分能力；
2. 将 `y_raw + scale-index` entropy 信号作为 innovation 驱动显式 SR state：有轻微增益但几乎不依赖正确时序，只是顺序不敏感的当前帧修正；
3. `ctx/ctx_t` 直接搬运（约 `192×46×80`，FP16 单张约 1.35 MiB）：导出与跨模型 tensor layout 成本过高，始终未采用；
4. 组合扫描 `ctx/ref_feature/router/loss` 已明确宣布不再尝试。

### 1.3 待决命题

第三阶段的最终命题（引用自原 08-27 文档）：

> MLVC 解码侧完整 latent `y_hat` 能否在极低部署成本下驱动可量化的 SR memory，使正确视频历史相对错误历史产生显著、稳定并且可见的超分收益。

## 2. 共同起点与视觉根因

原始部署基线：

| 实验                  | 阶段       | 最佳 step |    VMAF |    PSNR |
| --------------------- | ---------- | --------: | ------: | ------: |
| `phase-rlfn-codec-v1` | QAT stable |    52,000 | 70.6694 | 34.9340 |

三个阶段共同的视觉结论与根因：

- SR 相对 bicubic 的改善集中在轮廓、压缩伪影和局部对比度；没有出现栏杆、砖墙、文字笔画、织物等内部纹理的稳定恢复；
- preview 中 `SR EFFECT` 显示的是相对 HR 的误差改善，天然偏亮，不等于模型生成了新的高频内容；
- 根因不是某个 loss 或初始化选择：训练把 `B×T` 展平成独立帧，输出本质是"当前帧 bicubic + 局部残差"；MLVC decoder feature 即使含因果 memory，也位于低分辨率、由压缩目标训练、缺少未来帧与面向 SR 的显式历史传播；
- 打乱历史不掉分就意味着不是 VSR——这是贯穿三阶段的判伪标准。

## 3. 第一阶段（2026-08-24～25）：单帧局部改造，全部失败

围绕五类假设展开：recurrent memory 显式入参（memory_q）、phase residual 直接监督、逐帧误差 temporal consistency、可重参数化边缘分支、FP32→QAT 蒸馏。所有方案仍是"当前帧 bicubic + 局部残差"。

### 3.1 独立消融（2026-08-24）

| 实验             | 改动                                | FP32 VMAF | QAT stable VMAF | QAT PSNR |     相对基线 |
| ---------------- | ----------------------------------- | --------: | --------------: | -------: | -----------: |
| `memory-q`       | 仅 memory 48ch + normalized q plane |   70.2857 |         70.1793 |  34.8793 | -0.4901 VMAF |
| `phase-residual` | phase residual L1，权重 0.25        |   70.4059 |         70.1615 |  34.8786 | -0.5079 VMAF |
| `temporal`       | 相邻预测误差一致性，权重 0.10       |   70.1809 |         69.9529 |  34.8711 | -0.7165 VMAF |
| `reparam-edge`   | 3×3+1×1+1×3+3×1，随机初始化         |   68.8982 |         69.0730 |  34.7917 | -1.5964 VMAF |
| `qat-kd`         | FP32 teacher 蒸馏，权重 0.20        |   70.8875 |         70.7663 |  34.9308 | +0.0969 VMAF |

### 3.2 KD 权重扫描（相同 float 起点）

| KD 权重 | QAT observe VMAF | QAT stable VMAF |        PSNR |    相对 KD=0 |
| ------: | ---------------: | --------------: | ----------: | -----------: |
|     0.0 |          70.7449 |         70.6590 |     34.9278 |            — |
|     0.1 |          70.7972 |         70.7395 |     34.9302 | +0.0805 VMAF |
|     0.2 |      **70.8065** |     **70.7684** | **34.9302** | +0.1093 VMAF |

KD=0.2 能稳定恢复约 0.10 VMAF 的量化损失，但 PSNR 只高约 0.002 dB，不改变视觉性质——它是训练期量化保真技巧，不是新的 SR 信息路径；考虑 teacher/checkpoint/训练复杂度后不足以保留。

### 3.3 可复现 fresh-seed 主对照（single GPU，clip dropout）

| 实验                             |      FP32 VMAF / PSNR | QAT stable VMAF / PSNR | 判断                                |
| -------------------------------- | --------------------: | ---------------------: | ----------------------------------- |
| `codec-full-clip-control`        |     70.7827 / 34.9806 |  **70.4225 / 34.8900** | 同批最终最佳                        |
| `A5v2`（zero-init edge reparam） | **70.8531 / 34.9873** |      70.2310 / 34.8735 | FP32 微增 +0.0704，QAT 反转 -0.1915 |
| `codec-router`                   |     70.1970 / 34.9322 |      70.0687 / 34.8704 | FP32/QAT 均失败                     |

注意：该组是 single-GPU fresh-seed 对照，绝对分数不能与原始 2-GPU 基线或从共享 float checkpoint 启动的 KD 扫描混为一组。smoke 实验 `smoke-20260825-a5v2`、`smoke-20260825-codec-router` 仅验证链路可运行。

### 3.4 撤销记录（工作树）

撤销前状态：20 个已跟踪文件修改（878 insertions / 30 deletions）、11 个未跟踪 YAML 消融配置、无 staged 修改。训练产物在 git-ignored `checkpoints/` 中继续保留。

主要源码修改摘要（均已撤销）：

| 文件                                                  | 撤销前的修改内容                                                                                          |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `scripts/_common.sh`                                  | 支持 `TORCHRUN_MASTER_PORT` 指定并行实验 master port                                                      |
| `config/__init__.py` / `config/phase_rlfn_sr_x3.yaml` | 新增 seed、feature/dropout mode、reparam、router、辅助 loss、KD、梯度裁剪字段及默认值                     |
| `data/mlvc_loader.py`                                 | `full`/`memory_q`/`full_q` feature 模式、frame/clip dropout、独立 RNG 与可复现 seed                       |
| `data/mlvc_runtime.py`                                | 从 MLVC `compress_core` 暴露每帧 effective q index                                                        |
| `models/phase_rlfn_sr.py` / `models/__init__.py`      | 可融合 edge multi-branch conv、q-aware router、训练期 phase residual 辅助输出                             |
| `train/unified.py`                                    | 辅助 loss、QAT KD teacher、`--qat_from_float`、`--stop_after_phase`、seed/router 参数、checkpoint staging |
| `train/loop.py` / `train/types.py`                    | objective 接收辅助输出、loss component 与裁剪前 grad norm 记录、梯度裁剪配置                              |
| `utils/train_framework.py`                            | 全局 seed、AMP 安全梯度裁剪、新分支构造                                                                   |
| `deploy/onnx.py` / `deploy/rknn_eval.py`              | ONNX/RKNN 导出评估透传 reparam/router 配置                                                                |

配套测试修改：`test_config` / `test_mlvc_loader` / `test_mlvc_runtime` / `test_models` / `test_qat_utils` / `test_swanlab_logging` / `test_unified`。

11 个消融配置路径见附录 7.2。

### 3.5 不带回主线的方向

MLVC `memory_q/full_q` feature contract、effective-q 平面与 codec router、phase residual auxiliary loss、当前形式的 temporal error consistency、随机或 zero-init edge reparameterization、QAT teacher/staging/KD 框架，以及为它们增加的全部 CLI、配置、部署与测试表面。

### 3.6 若重启该方向，应先验证的一阶假设

失败说明应改变信息路径而非优化局部 residual：

1. 面向当前帧对齐的 MLVC `ctx/ctx_t`，而不是最终 `ref_feature`；
2. 低分辨率 causal SR recurrent state；
3. local/global 交错，让少数低成本全局/时序 mixer 承担长程信息；
4. 在相同部署预算下先证明"打乱历史会显著掉分"，再讨论感知 loss、MoE 或更复杂路由。

## 4. 第二阶段（2026-08-26）：IGR-PhaseSR entropy 门控，机制未通过

模型代号 **IGR-PhaseSR（Innovation-Gated Recurrent Phase SR）**。核心思想：不再把 MLVC 当作要塞进 SR 的 feature extractor，而是把神经码流看成逐帧的**预测误差信号与置信度观测**——SR state 保存已确认的低分辨率细节，当前 entropy symbol 表示新出现的 innovation，其编码代价与 scale-index 决定本帧保留多少历史、更新多少新信息。

### 4.1 设计思想（跨领域启发）

- **Kalman prediction–update**：不无条件覆盖历史，`next_state = prediction + gain × (candidate − prediction)`；gain 非严格概率意义，但承担相同职责——低创新区保留历史，高创新/不确定区积极更新；
- **Predictive coding**：码流中的 symbol 是"codec 预测之后仍无法解释的内容"，大 symbol、高 surprisal、高 scale-index 天然指向新结构/新纹理；
- **Event sensing**：只报告变化而非每帧重建全部历史，比搬运大尺寸 `ctx/ctx_t` 更符合 RK3588 带宽与算力预算；
- **Rate–distortion**：entropy bit cost 提供无需额外计算的"难预测性"先验。

### 4.2 信号来源与 98 通道契约

生产 decoder 在运行 RKNN 前，CPU rANS 路径已持有 `y_raw_0/1`（各 24×23×40）、scale-index `s0/s1`（各 24×23×40）与 rANS CDF 表，因此**不需要修改 MLVC decoder 图、不增加 NPU run**。`ctx/ctx_t/y_hat` 当时均在 RKNN 图内部不可得（这是后来第 5 章改用 decoder 第三输出的背景）。

98 通道构成：`y_raw_0/1` 各 24ch（`sign(y)·log2(1+|y|)/4` 归一化）、`s0/s1` 各 24ch（scale/127）、surprisal map 1ch、uncertainty map 1ch。部署侧 surprisal 用 `(scale_index, symbol)` 定点 LUT 实现，热路径不调 `log2`。

### 4.3 模型结构

单帧 core：输入 `phases 12×180×320`、`codec_signal 98×23×40`、`prev_state 16×45×80`；输出 `phase_residual 108×180×320`、`next_state 16×45×80`。空间主干沿用 Phase-RLFN（32ch、4 个双卷积残差块、低分辨率 phase domain、1×1 head），删除旧 `ref_feature → codec_expand → codec_fuse` adapter；`B×T` 在模型内按时间顺序逐帧递推。

Innovation 编码：`Conv1×1(64) → PixelShuffle×2 → 16×46×80 → crop → 16×45×80`；measurement 由图像 shallow 特征经两个 stride-2 3×3 卷积得到。

```text
prediction = Conv3x3(prev_state)
candidate  = Conv3x3(prediction + innovation + measurement)
gain       = Clip[0,1](Conv1x1(surprisal, uncertainty, prediction))
next_state = Clip[-8,8](prediction + gain * (candidate - prediction))
```

`next_state` 经 `Conv1×1 + PixelShuffle×4` 回到 `32×180×320` 加回主干。算子白名单仅 Conv/add/multiply/concat/Hardtanh/PixelShuffle，无 sigmoid、deformable、attention、Mamba、MoE 或生成式模块。

初始化：stem/blocks/feature_fuse/residual_head 载入既有最佳 FP32 checkpoint；旧 codec adapter 不载入；`state_to_main` 严格零初始化——step 0 输出与纯空间路径完全一致。

### 4.4 训练执行

8×A30 DDP，per-GPU sequence batch size=2；`sequence_frames=8`（1 个 DPB reference + 7 个监督 P-frame）；空间主干全程冻结，仅训练 recurrent 分支；Adam LR=1e-3；Charbonnier + 0.05× luminance 高频 DCT（仅 `u+v≥2` 系数）；state clip [-8,8]；无 codec dropout；每 500 step 记录、每 1,000 step 固定 16 样本验证；`--float-only`，绝不自动进入 QAT。正式意图中后续的 Stage B（联合训练）/ Stage C（QAT）因 Stage A 未过门控而从未启动。

`igr-20260826-entropy-gate-v1` 于 8×A30 正常完成 10,000 step，最佳 checkpoint 位于 step 8,000：

|   step |        VMAF |        PSNR |
| -----: | ----------: | ----------: |
|  1,000 |     70.4824 |     34.9738 |
|  5,000 |     70.5553 |     34.9798 |
|  8,000 | **70.5755** |     34.9812 |
| 10,000 |     70.5752 | **34.9830** |

### 4.5 判伪：matched 对照与历史消融

相同 16 个验证样本、相同空间权重、best checkpoint（step 8,000）：

| 模式             |        VMAF |        PSNR | 相对正确历史 |
| ---------------- | ----------: | ----------: | -----------: |
| 初始冻结空间模型 |     70.4825 |     34.9534 |      -0.0934 |
| 正确历史         | **70.5759** | **34.9812** |            — |
| 每帧 reset state |     70.5141 |     34.9776 |      -0.0618 |
| 前序历史倒序     |     70.5723 |     34.9801 |  **-0.0036** |

事实判断：

1. 相对 matched spatial control 仅 +0.0934 VMAF，低于预设直接停止线 +0.10；
2. reset 损失 0.0618 说明 state 不是完全无效，有轻量的时间累积作用；
3. 倒序仅损失 0.0036，远低于 0.15 的最低要求——模型学到的是时间累积、平滑或顺序不敏感的当前帧修正，没有建立可验证的因果历史利用。

预期中的对照关系在此印证："若分数提升但 shuffle 几乎不掉分，只能说明 entropy signal 是有效的当前帧 quality/saliency prior，不能证明获得 VSR 能力"。

### 4.6 饱和与量化风险

- gain `<0.01` 占 13.71%、`>0.99` 占 46.33%（两极饱和）；
- state 触及 ±8 边界的比例 6.71%；
- `innovation_expand` 权重范数 84.13、最大绝对值 4.50，而后续 QAT 权重裁剪范围为 [-1,1]，小幅增益大概率无法保留。

结论处置：不解冻、不续训、不做 QAT。定位为轻量 compressed-video enhancement，不作为下一代 VSR 主线。

已知风险的事后印证（当初预判，现均成立）：熵 signal 只反映压缩难度的风险、state 被逐帧捷径绕过的风险，均指向"机制门槛必须靠 history shuffle 硬性验收"这一判断的正确性。

## 5. 第三阶段（2026-08-27）：最终 `y_hat` 判伪实验（中断，结论未定）

### 5.1 为什么最后尝试 y_hat

`y_raw` 只是打包后的量化残差，缺少 entropy mean、空间先验恢复结果和实际反量化尺度；`y_hat` 则是：

```text
y_hat = (unpack(symbol_0 + mean_0) + unpack(symbol_1 + mean_1)) * q_dec
```

它仍在紧凑的 codec bottleneck 内，却比裸 symbol 更接近 MLVC decoder 真正使用的时空表示，且是唯一仍有明确一阶信息差异、部署成本可接受的候选。`ctx/ctx_t` 因体量与 layout 风险被排除，旧 `ref_feature` 已被第一阶段消融否决。唯一预先约定的主救援协议：替换 innovation 输入，其余不变；若 `y_hat` 仍不能建立 history dependency，codec-aware recurrent VSR 主线结束。

### 5.2 输入契约与固定归一化

```text
normalized_y_hat : 48 × 23 × 40
surprisal_map    :  1 × 23 × 40
uncertainty_map  :  1 × 23 × 40
--------------------------------
codec_signal     : 50 × 23 × 40
```

surprisal/uncertainty 两张图只参与 gain 的显式置信度输入；`y_hat` 整体替换旧的 96 通道 symbol/scale 部分，不与旧输入混叠，保证归因清晰。模型改动仅在 codec 输入通道 98→50。

固定归一化统计自 8 个确定性 OpenVidHD 训练 clip、四个 q-index、共 224 个 P-frame 的逐通道 `abs(y_hat)` P99.9：

- 通道尺度范围 0.9218～13.5232，全局 P99.9 = 4.2838；
- 逐通道缩放并裁剪到 [-1,1] 后实际裁剪比例约 0.1004%；
- 常量位于 `data/mlvc_runtime.py`，禁止逐帧动态归一化；部署时缩放可与 RKNN 输入量化合并，不增加 MLVC NPU run。

训练侧 `compress_core()` 已直接返回 `result["y_hat"]`，科学验证不依赖 SDK 修改。

### 5.3 SDK 边界与算力预算

只有实验通过后才给 MLVC decoder RKNN 增加第三输出 `y_hat`（`48×23×40`，FP16 约 86.25 KiB/frame、INT8 约 43.13 KiB/frame）：不拆分 decoder、不增加第二次 MLVC run、在验证 RKNN tensor attr 前不假设跨模型 native zero-copy。

| 项目              | entropy IGR（二阶段） | `y_hat` IGR（三阶段） |      预算 |
| ----------------- | --------------------: | --------------------: | --------: |
| 参数量            |               111,604 |               108,532 |  ≤150,000 |
| Core MAC          |           5.0083 GMAC |        约 5.0055 GMAC | ≤5.1 GMAC |
| MLVC 额外 NPU run |                     0 |                     0 |         0 |
| SR NPU run        |               1/frame |               1/frame |   1/frame |

显存实测：单卡 batch=2、7 P-frame 前向反向峰值约 3.71 GiB；8 卡正式运行约 6.3～6.4 GiB/GPU（A30 24 GiB 余量充足）。部署预期（继承自 IGR 设计）：SR 总延迟 ≤ 旧模型 1.10×、host 端 98→50 通道信号构造不得成为 CPU 瓶颈、300 帧连续运行无 drift、seek/丢帧/I-frame/场景切换确定性清零 state。

### 5.4 原定训练计划

全新 recurrent 分支（不继承 entropy IGR 已饱和的权重）：空间起点 `checkpoints/hf-package/phase-rlfn-codec-v1/float/best.pth`；8×A30 DDP、per-GPU sequence bs=2、`sequence_frames=8`（1 DPB + 7 监督 P-frame）；空间主干在全部 10,000 update 内冻结；Adam，recurrent LR=1e-3；Charbonnier + 0.05× luminance 高频 DCT；state clip [-8,8]；无 codec dropout；每 500 step 记录、每 1,000 step 验证；step 5,000/10,000 存 checkpoint；`--float-only`。不加 history ranking——门控阶段不能用辅助目标强迫无效信号表演历史依赖。

### 5.5 当前真实状态（2026-08-27）

```text
experiment: igr-20260827-yhat-final-v1
checkpoint: checkpoints/igr-20260827-yhat-final-v1/
SwanLab:    https://swanlab.cn/@Sail2Dream/rknn-super-resolution/runs/u5xafqd3
tmux 会话:  igr-yhat-final-20260827（已不存在）
```

- 训练于 UTC 约 02:33 启动，8 rank 正常，`torch.compile`、1-step smoke、SwanLab ping/verify 均通过；
- UTC 约 02:56 进程收到 SIGINT 手动中断，止步于 step ≈5,500；
- 日志末尾的验证：step 5,000 val VMAF≈70.57 / PSNR≈34.98 / SSIM≈0.9421，loss 稳定在 ~1.54（charb ~1.38、dct ~0.161）；
- 已保存 best.pth 与 step_5000.pth；step 10,000 未到达；
- 5.6 节的四组验收评估尚未执行，**最终判伪悬置**。

重启或补完本实验时应沿用同一配置 `src/rknn_super_resolution/config/igr_y_hat_phase_sr_x3.yaml` 与 5.6 节门槛，不得中途放宽。

### 5.6 不可修改的验收门槛

训练后在相同验证集运行 spatial control、正确历史、reset state、倒序历史四组评估，同时满足才进入联合训练：

| 诊断                                 |                                       通过线 |
| ------------------------------------ | -------------------------------------------: |
| 正确历史相对 matched spatial control |                                 ≥ +0.25 VMAF |
| 倒序历史相对正确历史                 |                                 ≤ -0.15 VMAF |
| reset state                          |                               稳定、明显下降 |
| PSNR                                 |   下降不超过 0.03 dB，除非有明确真实纹理收益 |
| state clipping                       |                                  目标低于 1% |
| 视觉                                 | 改善进入内部纹理，无稳定 ghosting/闪烁/drift |

+0.10～+0.25 VMAF 区间只能说明可能是 codec enhancement；不足以支付 recurrent state 和 SDK 改造成本，不进入 QAT。entropy IGR 的 +0.0934 即栽在最初的 +0.10 线上。

### 5.7 两个终局

**若通过**：修改 decoder RKNN 增加 `y_hat` 第三输出并实测 handoff 成本 → sequence 扩展到 16、解冻空间主干 → 10% batch 加入 correct-vs-shuffled history ranking → FP32 通过后做 QAT（recurrence 在图内、`prev_state/next_state` 固定量化范围、单独检查 scale/zero-point，增益保留率目标 ≥ FP32 增益的 70%）→ RK3588 连续 ≥300 帧验证 seek/丢帧/I-frame/场景切换 reset。

**若失败**：宣布 codec-aware recurrent VSR 主线结束；不再尝试 `ctx/ref_feature/router/loss` 组合扫描；产品线回到单帧 Phase-RLFN/QAT（以量化稳定与真实延迟为目标）；若仍研究视频超分，另立显式运动对齐项目，不伪装成本分支小改动。

## 6. 工程验证记录（跨阶段累计）

以下均为已完成的工程事实，非预期：

- MLVC-S 同一次 `compress_core` 可直接取得 codec signal（entropy 版 98ch、`y_hat` 版 50ch），Python 训练路径无额外 MLVC forward；decoder feature 在 IGR 配置中不再堆叠返回；
- `B×T` 顺序在模型内保留、state 逐帧传递；训练模式返回所有 P-frame、验证模式返回最后一帧；
- FP32 baseline 空间层全部匹配载入；零初始化保证新增 state 分支不改变纯空间输出；
- `torch.compile` 前向反向通过；1-step 真实数据 smoke train/validation/checkpoint 全链路（两轮实验各自通过）；
- 测试规模随实现演进：135 passed（IGR 实现）→ 138 passed（`y_hat` 数据路径落地），ruff 通过；
- SwanLab 网络/登录验证通过（账号 Sail2Dream）；首轮上传期间本机代理 `127.0.0.1:7890` 曾短暂不可达后自动恢复并显示 Upload complete。

## 7. 附录

### 7.1 checkpoints 目录索引

```text
checkpoints/phase-rlfn-codec-v1/                    # 原始基线
checkpoints/ablation-20260824-memory-q/
checkpoints/ablation-20260824-phase-residual/
checkpoints/ablation-20260824-qat-kd/
checkpoints/ablation-20260824-reparam-edge/
checkpoints/ablation-20260824-temporal/
checkpoints/ablation-20260825-kd-{000,010,020}/
checkpoints/ablation-20260825-codec-full-clip-control/
checkpoints/ablation-20260825-a5v2/
checkpoints/ablation-20260825-codec-router/
checkpoints/smoke-20260826-igr/
checkpoints/hf-package/phase-rlfn-codec-v1/float/best.pth   # 三阶段复用的空间起点
checkpoints/igr-20260826-entropy-gate-v1/           # 二阶段（已完成）
checkpoints/igr-20260827-yhat-final-v1/             # 三阶段（中断）
```

### 7.2 第一阶段消融配置（已随代码一并撤销）

```text
config/ablations/20260824_memory_q.yaml          config/ablations/20260825_codec_full_clip_control.yaml
config/ablations/20260824_phase_residual.yaml    config/ablations/20260825_codec_router.yaml
config/ablations/20260824_qat_distillation.yaml  config/ablations/20260825_qat_kd_{000,010,020}.yaml
config/ablations/20260824_reparam_edge.yaml      config/ablations/20260825_reparam_edge_v2.yaml
config/ablations/20260824_temporal_consistency.yaml
```

### 7.3 SwanLab 运行索引

| 实验                    | 链接                                                                     |
| ----------------------- | ------------------------------------------------------------------------ |
| codec-full-clip-control | https://swanlab.cn/@Sail2Dream/rknn-super-resolution/runs/7fp6qwig/chart |
| A5v2                    | https://swanlab.cn/@Sail2Dream/rknn-super-resolution/runs/3h8u63ym/chart |
| codec-router            | https://swanlab.cn/@Sail2Dream/rknn-super-resolution/runs/j965gw7u/chart |
| KD=0.0 / 0.1 / 0.2      | runs/2185hcsb · runs/0oyqfcje · runs/31526yam                            |
| IGR entropy-gate-v1     | https://swanlab.cn/@Sail2Dream/rknn-super-resolution/runs/b115zks6       |
| y_hat final v1          | https://swanlab.cn/@Sail2Dream/rknn-super-resolution/runs/u5xafqd3       |

## 8. 最终判断原则

整个研究方向不是为了寻找一个更高的单点 VMAF，而是反复验证同一个强命题：

> 神经码流的信息能否以极低部署成本控制 SR memory 的保留与更新，让正确的视频历史产生可测、可视、可量化保留的超分收益。

分数提升、history shuffle 显著退化、长序列无漂移、QAT 保留大部分增益——四者依次成立才算成功；任何阶段失败都不得用更大主干、更强感知 loss 或更多 adapter 掩盖。
