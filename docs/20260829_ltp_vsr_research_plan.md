# CST-VSR 有效性审计与修订研究方案

- 审计日期：2026-08-29（UTC）
- 原方案：LTP-VSR（Lattice-Transport Phase VSR）
- 修订候选：**PST-VSR（Polyphase Splat Transformer VSR）**
- 当前结论：**原 LTP-VSR 与 coarse-value CST 均否决；PST-VSR 已通过合成信息、坐标保真和 RKNN 稀疏注意力微图门槛，尚无真实视频质量结果**
- 继承材料：[`20260827_codec_aware_vsr_research_archive.md`](20260827_codec_aware_vsr_research_archive.md)

## 1. 结论先行

上一版方案的两个核心假设经直接分析后不能成立：

1. `PixelUnshuffle(2)` 只是 LR 网格的奇偶位置重排，不是 3× SR 所缺失的 HR 子像素观测。普通 `Shift(prev_state)` 不能解释成“相位亚像素传输”；
2. 13 候选全分辨率 alignment 虽然参数和 MAC 很小，但 RKNN 编译器显示其内存流量和内部张量远大于空间基线，完整模型无法可信地满足 1.25× 延迟预算。

旧 `y_hat` recurrent checkpoint 的权重级分析还表明，它不是内在稳定的递推系统：零外部输入附近的局部状态 Jacobian 最大奇异值约为 `4.735`，无外部观测递推时很快依赖 `Clip[-8,8]` 限幅。这与训练日志中的 gain 两极饱和和 state clipping 一致。

因此本轮不再为 LTP-VSR 辩护。修订候选 CST-VSR 做三项根本改变：

- 去掉“phase transport”声明，只主张**稀疏离散对齐**；
- alignment 从 `180×320` 下沉到 `45×80`，状态从约 1.32 MiB 降至 112.5 KiB；
- 把观测 key 与递推 detail 分离，令 detail 更新在结构上满足 `L∞` 收缩上界，而不是靠末端 clip 防爆。

这仍不是有效性证明。当前最大的未解问题是：`45×80` coarse state 加 nearest injection 是否还保有足够的细运动和高频信息。它必须先通过 oracle upper-bound 与合成位移门控，才值得进行真实 MLVC 训练。

## 2. 证据边界

本文件区分三类证据：

- **直接测量**：checkpoint 权重、ONNX 图、RKNN Toolkit2 2.3.2 编译日志和模拟器输出；
- **数学结论**：由写明的张量映射和约束推出；
- **待验证假设**：真实视频质量、真机延迟、训练可学性和“划时代性能”。

本轮没有连接 RK3576/RK3588 真机，因而 compiler cycles 不是 wall-clock latency；PC simulator 只证明图可执行和输出有限值。临时 ONNX、RKNN、校准数组和日志位于 `/tmp/ltp_rknn_probe/`，不作为版本库产品产物。

## 3. 小模型直接审计

### 3.1 规模与算力

| 对象 | 直接测量 |
| --- | ---: |
| 当前 Phase-RLFN 全模型参数 | 110,252 |
| 当前部署 core Conv MAC | 5.00072448 GMAC/frame |
| `y_hat` IGR checkpoint 总参数 | 108,532 |
| 其中 recurrent 分支参数 | 23,368 |
| 其中空间主干参数 | 85,164 |
| `y_hat` checkpoint 内部 best VMAF / PSNR | 70.5676763 / 34.9835321 dB |

模型足够小，不能再以“黑盒太复杂”为由只看最终 VMAF。权重谱、局部 Jacobian、逐通道 lesion、状态扰动和全候选消融都应成为标准诊断。

### 3.2 `y_hat` recurrent cell 的稳定性

按归档中的递推式重建状态路径：

```text
prediction = Conv3x3(prev_state)
candidate  = Conv3x3(prediction + innovation + measurement)
gain       = Clip[0,1](Conv1x1(surprisal, uncertainty, prediction))
next_state = Clip[-8,8](prediction + gain * (candidate - prediction))
```

| 检查 | 结果 | 含义 |
| --- | ---: | --- |
| `state_predict` 圆周卷积频域最大增益 | 2.0225 | prediction 不是非扩张映射 |
| `state_predict` 最大增益 > 1 的频点比例 | 73.94% | 放大不是单一异常频点 |
| `state_candidate` 圆周卷积频域最大增益 | 6.4924 | candidate 路径放大更强 |
| `state_candidate` 最大增益 > 1 的频点比例 | 100% | 所有检查频点均可能放大 |
| 零外部输入附近局部 Jacobian `σmax` | 4.7353 | 局部不是收缩映射 |
| `state_to_main` 矩阵秩 / 条件数 | 16/16 / 9.016 | 非简单死通道，但不保证有用 |

在“外部 innovation、measurement、gate map 全零”的条件压力测试中，从全零 state 连续递推：

| 帧 | state clip 比例 | `1e-6` 初始扰动后的两轨迹差范数 |
| ---: | ---: | ---: |
| 8 | 11.94% | 0.000581 |
| 30 | 17.73% | 0.03091 |
| 60 | 18.26% | 0.5758 |
| 120 | 17.81% | 13.9173 |
| 300 | 17.77% | 19.5107 |

这不是“真实视频必然发散”的证明，因为真实 measurement 会改变动力系统；但它证明旧 cell 没有内在稳定保证，`Clip[-8,8]` 是实际动力学护栏而非偶尔触发的安全网。

### 3.3 量化前风险

默认配置每步把权重裁剪到 `[-1,1]`。`y_hat` checkpoint 的 `innovation_expand` 有 39.16% 权重超界；硬裁剪造成权重相对 RMSE 44.67%，随机归一化输入下该层输出相对 RMSE 44.62%。以后进入 QAT 前必须报告逐层 clip sensitivity，不能假设微小 FP32 增益会自然保留。

## 4. “相位传输”假设的直接否证

设 `P_r` 为 `PixelUnshuffle(r)`，`T_Δ` 为 LR 图像整数平移。对 `r=2`、水平平移 1 像素：

```text
P₂(T₁ x)[c, py, px, i, j]
= P₂(x)[c, py, (px-1) mod 2, i, j + floor((px-1)/2)]
```

它需要 phase channel permutation 和由输出 phase 决定的不同 coarse-grid shift。固定随机张量的直接反例：

| 变换 | 相对 RMSE |
| --- | ---: |
| 最佳整块 phase tensor 空间平移 | 1.398286 |
| 精确 phase permutation + phase-specific shift | 0.0 |

更重要的是，`PixelUnshuffle(2)` 的四组通道来自已观测 LR 像素的奇偶位置，而 3× SR 要恢复的是每个 LR 像素内部未观测的 HR 子位置，两者不是同一个 phase。

结论：**LTP-VSR 按原机制说明 REJECT**。固定 shift mixture 可以继续作为离散空间重采样，但不得再包装成已成立的亚像素相位对齐。

## 5. RKNN 工具链实测

环境为 RKNN Toolkit2 2.3.2、隔离 Python 3.12、INT8 per-channel PTQ。初次 setup 时，GUI 版 `opencv-python` 因服务器缺少 `libGL.so.1` 导致 RKNN import 失败；本轮在隔离环境内用同版本 headless OpenCV 文件替代，并保留 toolkit 所需包元数据，`uv pip check` 与 RKNN import 均通过。此环境问题与候选算子支持无关。

下表的 NPU/CPU 数量排除 Input/Output operator；cycles 与 RW 是编译表逐层求和，只用于同一工具链的相对比较。

| 图 | target | 计算 placement | total cycles | 累计 RW | 内部内存 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 当前 Phase-RLFN codec core | RK3576 | 16 NPU / 0 CPU | 16.460979M | 52.759 MB | 18.000 MB | 比较基线 |
| Phase-RLFN image-only anchor | RK3576 | 12 NPU / 0 CPU | 8.277243M | 47.784 MB | 18.900 MB | 隔离 codec adapter 成本的公平对照 |
| 原 13 候选 full-res | RK3588 | 89 NPU / 1 CPU | — | — | 51.323 MB | 8ch guide `Pad` 落 CPU |
| 因式分解 13 候选 full-res | RK3588 | 91 NPU / 0 CPU | — | — | 52.254 MB | 全 INT8/NPU，但内存大 |
| 因式分解 13 候选 full-res | RK3576 | 91 NPU / 0 CPU | 3.827595M | 130.154 MB | 51.354 MB | alignment 已用基线 23.25% cycles，RW 是 2.47× |
| 13 候选 coarse alignment | RK3576 | 88 NPU / 0 CPU | 0.200902M | 7.935 MB | 3.113 MB | 可行 |
| full branch + `ReduceMax` | RK3576 | 99 NPU / 1 CPU | 0.675553M | 11.330 MB | 4.120 MB | `ReduceMax` 落 CPU，否决 |
| 中心 score 复用 | RK3576 | build failure | — | — | — | optimizer `KeyError: mul_1_rs` |
| full branch + PixelShuffle×4 | RK3576 | 106 NPU / 0 CPU | 15.700602M | 20.822 MB | 11.025 MB | 降成两路 ConvTranspose，否决 |
| **CST full branch V4** | **RK3576** | **103 NPU / 0 CPU** | **0.721978M** | **13.467 MB** | **4.176 MB** | **当前唯一保留候选** |
| **CST + Phase-RLFN 整图 V2** | **RK3576** | **116 NPU / 0 CPU** | **8.999221M** | **63.051 MB** | **18.689 MB** | **整图编译门槛通过** |

实测给出了官方 op-support 表无法回答的事实：`Softmax(K=13)` 会编译成 INT8 NPU `exSoftmax13`；同类 `Pad` 的 placement 受通道 layout 影响；`ReduceMean` 被改写为 13 个 1×1 Conv；PixelShuffle×4 的 ConvTranspose lowering 成本远高于朴素 MAC 直觉。

整图相对 image-only anchor 的 compiler cycles 为 `1.0872×`、累计 RW 为 `1.3195×`，通过预注册的 `1.10×` / `1.35×` 门槛；内部内存未增加。相对当前 codec baseline，整图 compiler cycles 为 `0.5467×`，说明 codec adapter 的 PixelShuffle/ConvTranspose lowering 是当前部署图的重要成本，而不是 CST 免费。

整图 simulator 三输出均为有限值；`next_key` 与 `candidate` 显式裁剪到 `[-1,1]`，`next_detail` 由 bounded history 与 candidate 的凸组合保持同一不变量，两个状态输出的量化范围实测均约为 `[-0.9961,0.9961]`。真机 latency 仍未验证，compiler cycles 不能替代它。

## 6. 修订模型：CST-VSR

### 6.1 状态与计算图

```text
current shallow : 32 × 180 × 320
prev_key        : 16 ×  45 ×  80   # 上一帧观测缓存，不递推融合
prev_detail     : 16 ×  45 ×  80   # 唯一递推记忆

m_t        = LeakyReLU(Conv3x3,s2(LeakyReLU(Conv3x3,s2(shallow_t))))
key_t      = Clip[-1,1](Conv1x1(m_t))
score_i    = MeanChannel(key_t * Shift(prev_key, offset_i))
alpha      = Softmax(score, candidate_axis)
aligned    = Sum_i alpha_i * Shift(prev_detail, offset_i)
confidence = Conv1x1(Concat(key_t, prev_key))
candidate  = Clip[-1,1](Conv3x3(m_t))
gate       = 1/8 + 7/8 * HardSigmoid(Conv1x1(m_t, confidence))
next_detail = (1 - gate) * aligned + gate * candidate
next_key    = key_t
temporal_inject = NearestResize×4(Conv1x1(next_detail))
```

`temporal_inject` 在 stem 后、残差块前加入主干；正式模型的 project 权重零初始化，使 step 0 与 spatial anchor 相同。微图使用非零探针权重，避免 ONNX 常量折叠输出。

INT8 state 为 `32×45×80 = 115,200` bytes，即 112.5 KiB。修订 branch 约 20,121 参数；移除 codec adapter 后合计约 105,285 参数，Conv MAC 约增加 0.147 GMAC。参数量不是主要约束，NPU layout 与 DDR 才是。

### 6.2 收缩性

对固定的 current shallow 与 prev key，`alpha` 与 prev detail 无关。令 `T_i` 为零填充 shift：

```text
A(d) = Sum_i alpha_i T_i(d)
||A(d1)-A(d2)||∞ ≤ ||d1-d2||∞
```

`candidate` 和 `gate` 也不依赖 prev detail，且 `gate ≥ ε = 1/8`，所以：

```text
Φ(d) = (1-gate)A(d) + gate*candidate
||Φ(d1)-Φ(d2)||∞ ≤ 7/8 ||d1-d2||∞
```

这是结构保证，不需要假设训练后卷积谱范数自动小于 1。prev key 每帧由当前观测覆盖，不形成另一条自由递推路径。

INT8 simulator 上以固定 shallow/key、24 组随机 detail 扰动检查，观测 next-detail `L∞` 增益最大 0.3279，next-key 对 detail 扰动为 0。该结果是有限样本 sanity check，不替代所有 INT8 输入上的形式证明。

## 7. 当前最强反论点

> CST-VSR 可能只是一个稳定、便宜、可部署，但无能力恢复细节的低分辨率时间平滑器。

这个反论点目前成立：alignment 在 `45×80`，一个 coarse shift 对应 4 个 phase-domain 像素；nearest ×4 本身不产生高频；正确历史依赖、亚像素可辨识性和真实 MLVC 增益均未测。“稳定 + 可编译”只把候选从不可行区移入可实验区。

## 8. 下一阶段：先测信息上界

### Stage 0B：整图 RKNN（compiler gate 已通过）

V4 已与真实 Phase-RLFN 合图并使用非零 probe project。RK3576 结果为：

- 116 个 INT8 计算层全部在 NPU；
- compiler cycles 为 image-only anchor 的 `1.0872×`；
- 内部内存 18.689 MB，累计 RW 为 anchor 的 `1.3195×`；
- simulator 的三个输出均为有限值。

compiler gate 判定 **PASS**。仍需在真机可用后测 p50/p90，最终 core latency 必须 ≤ 1.25×。

### Stage A0：oracle upper bound

用已知位移生成受控序列：HR 纹理经过 `{-1.5,-1.0,-0.5,0,0.5,1.0,1.5}` LR pixel 位移、下采样与 MLVC 风格退化。比较 spatial anchor、oracle full-resolution warp、oracle coarse transport、uniform/center/wrong-offset。

若 oracle coarse transport 相对 spatial 的上界低于 `+0.50 VMAF` 或 `+0.10 dB`，CST 直接终止；训练不能创造输入路径本身没有的可辨识信息。

### Stage A1：穷举诊断

只训练约 20k temporal 参数，每 500 step 执行：

- 13 个 candidate 逐一删除与 one-hot 强制；
- 16 个 detail channel 逐一置零；
- correct/reset/reverse/wrong-clip history；
- alpha entropy、gate 分布、state 范围；
- FP32 状态 Jacobian power iteration；
- 每层权重超界比例与 clip-output sensitivity；
- INT8 扰动增益复测。

任何增益若只来自静态 channel 或对历史顺序不敏感，都按单帧去压缩器处理，不进入联合训练。

### Stage B：真实 MLVC 门控

冻结空间主干，真实 OpenVidHD/MLVC-S 训练。5k step 必须同时满足：correct 相对 spatial ≥ `+0.35 VMAF`，相对 reset ≥ `+0.25`，reverse 相对 correct ≤ `-0.20`，wrong-clip ≤ `-0.30`，三个 seed 方向一致并报告 paired bootstrap CI。未过门槛不解冻主干，也不进入 QAT。

## 9. “划时代性能”的冻结定义

只有最终 QAT 模型同时达到以下条件，才允许使用强性能表述：

| 维度 | 通过线 |
| --- | ---: |
| QAT VMAF 相对 `phase-rlfn-codec-v1` | ≥ +1.00 |
| QAT PSNR | ≥ +0.10 dB，或盲评有明确纹理收益且不低于 -0.03 dB |
| correct history 相对 reset | ≥ +0.50 VMAF |
| reverse/wrong history 相对 correct | ≤ -0.30 VMAF |
| spatial-width-control | CST ≥ +0.35 VMAF |
| QAT 时间增益保留率 | ≥ 85% |
| 真机 core latency | ≤ 1.25× baseline |
| 3,000 帧 | 无发散、稳定 ghosting、闪烁或 state drift |

## 10. 审计判决

| 问题 | 判决 |
| --- | --- |
| 原 LTP phase 解释是否成立？ | **REJECT** |
| 原 full-resolution 设计是否符合部署预算？ | **REJECT** |
| 旧 y_hat cell 是否有内在稳定保证？ | **REJECT** |
| RKNN 是否支持 shift + Mul + Softmax + weighted sum？ | **PASS，受通道/layout 条件约束** |
| CST V4 branch 与 Phase-RLFN 整图是否可全 INT8/NPU 编译并运行 simulator？ | **PASS** |
| CST 是否已证明提升真实视频质量？ | **NO EVIDENCE YET** |
| 下一项最高信息价值实验 | **oracle coarse-transport upper bound** |

## 11. 研究诚信与复现限制

- compiler cycles、RW 和 simulator 不是 RK3576 真机 latency；
- 随机 calibration 只验证算子、placement 与图路径，不用于声明量化精度；
- 旧 cell 压力测试以归档公式为准；缺少已删除训练源码时，不冒充完整 end-to-end 复现；
- coarse-state 功效、history dependency、QAT 保留率均未决；
- 超分细节是模型估计，不应用于取证式身份或事实判断；
- 本文件由 AI 辅助分析与写作，所有数字来自本地 checkpoint、源码、ONNX/RKNN 日志或明确计算，不由目标结论反推。

## 12. Stage A0 实验结果：CST value path 否决

本轮补做了三种子、同 seed 数据与初始化配对的 5k-step 小模型探针。最初的 LR-layout proxy 不是 Phase-RLFN 的真实张量布局，只作为探索数据保留；正式判决使用 `PixelUnshuffle(2)` 后的 `180×320` phase layout。

| phase-layout value path | 相对 spatial PSNR，3 seeds | 判决 |
| --- | ---: | --- |
| center 容量对照 | `+0.0020 ± 0.0020 dB` | 无时间信息 |
| full `180×320`，LR 双线性对齐 | `+0.0211 ± 0.0059 dB` | FAIL |
| half `90×160`，平均压缩 | `+0.0030 ± 0.0011 dB` | FAIL |
| coarse `45×80`，平均压缩 | `+0.0017 ± 0.0020 dB` | **FAIL，终止 coarse-value CST** |

`45×80` 与 center 的差为 `-0.0003 ± 0.0005 dB`。这个结果否决的是“LR warp → coarse average/detail → nearest injection”路径，不是否决多帧信息本身。原因审计表明，LR 双线性对齐会把多帧超分所需的亚像素采样相位先行混合掉，因此它不构成信息论上界。

## 13. 拆分数据上界与网络上界

### 13.1 已知成像方程的直接反演

新增 target-aware 迭代反演作为慷慨的信息上界：空间版本只拟合当前 LR 观测，时间版本同时拟合当前与四帧已知位移历史。3 seeds × 4 batches、每 batch 4 个样本的结果：

| 指标 | 结果 |
| --- | ---: |
| temporal 相对 spatial inverse PSNR | `+1.2914 ± 0.6484 dB` |
| batch 方向一致性 | `12/12` 为正 |
| 高频 PSNR 差 | `+0.2141 ± 0.2003 dB`，`11/12` 为正 |
| 最小 / 最大 PSNR 差 | `+0.5149 / +2.4643 dB` |

目标参与 iterate 选择，所以这不是可实现模型成绩；它只证明原始观测中确有显著可恢复信息。视觉面板显示 bicubic/单帧反演丢失的斜向周期纹理在多帧反演中部分恢复，同时保留了明显反演伪影，符合 upper-bound 而非产品输出的定位。

### 13.2 HR 坐标对齐的跨帧注意力

历史观测先 bicubic 到 HR 坐标、按已知运动注册，再用当前 query 对历史 key/value 做帧轴 Softmax。相对同容量 spatial：

- attention：`+0.2775 ± 0.0364 dB`；
- 简单 registered mean：`+0.2767 ± 0.0355 dB`；
- attention 相对 center：`+0.2765 ± 0.0380 dB`；
- 错误运动方向相对正确方向损失约 `0.93–1.20 dB`。

这证明坐标正确时普通小网络能使用历史，但 Softmax 本身只贡献约 `0.001 dB`。决定性因素是坐标/采样保持，不是“Transformer”标签。

## 14. Polyphase splat：首个强机制结果

对半 LR 像素位移，先把每个 LR 样本放到 `2×LR` 稀疏观测格，再做整数坐标注册；随后用 `PixelUnshuffle(2)` 把四个真实采样 phase 与四个 observation-count mask 打包回 LR。整个过程不对历史观测值做双线性混合。

同容量 32-channel/4-block/5k-step 探针结果：

| Seed | Spatial | Polyphase splat | Δ |
| ---: | ---: | ---: | ---: |
| 20260829 | 18.8566 | 20.0573 | `+1.2007` |
| 20260830 | 19.0835 | 20.2294 | `+1.1459` |
| 20260831 | 19.1596 | 19.9978 | `+0.8382` |
| **均值** | — | — | **`+1.0616 ± 0.1954 dB`** |

center 与 spatial 完全相同；正确坐标相对 wrong-coordinate 的单模型消融差约 `6.2 dB`。这是当前最强的机制证据：细节必须沿 observation phase 传输，不能沿 coarse averaged feature 传输。

合成位移并不难识别。对 36 个二维半像素候选做无需训练的 LR 相关搜索，在 192 个历史配对上 top-1 为 `100%`、offset MAE 为 `0`。因此下一模型可把 attention 用于坐标后验，而让 value path 执行 polyphase splat。

进一步把真值位移从训练输入中移除：每个 batch 只用 current/history LR 做 36 候选相关检索，hard top-1 坐标再驱动同一 splat。三种子相对 spatial 为 `+1.0963 ± 0.1479 dB`（`+1.2053/+1.1556/+0.9279`），与 oracle-coordinate splat 的 `+1.0616 ± 0.1954 dB` 在训练方差内等价。合成域的检索→坐标→value transport 因果链成立。

hard global argmin 不是可部署 PST：它利用了合成序列的全局刚性运动，也不提供遮挡/低置信区域的连续 posterior。下一步仍必须替换为局部、可量化 Softmax，并以真实 MLVC 质量验证。

## 15. 真实 MLVC 局部相关观测

在安装 FFmpeg 8 并确认 TorchCodec/NVDEC 无 CPU fallback 后，对 16 个独立 validation clips、`q={0,21,42,63}`、每段 7 个历史 P-frame 做 8×8 LR block correlation，共 64 个 clip-quality 组合。它仍属于结构筛选而非正式质量声明。

| 候选 bank | block MSE 相对 center 降低 | 非中心选择率 | 选择熵 |
| --- | ---: | ---: | ---: |
| K=5 integer cross | `31.00 ± 10.22%` | `68.65%` | `0.830` |
| K=13 half-pixel cross + diagonal | `40.16 ± 11.16%` | `80.79%` | `0.810` |
| K=25 half-pixel grid | `44.73 ± 11.85%` | `80.99%` | `0.780` |

K=13 取得 K=25 可见 error-reduction 的约 89.8%，而可视化的 offset/confidence/improvement 区域与物体和边缘结构一致。K=5 更便宜但覆盖不足；K=25 的额外 4.57 percentage points 暂不足以为更大候选张量辩护。当前默认保留 K=13，并要求后续用 quality-vs-RW 实验重新确认。

## 16. RKNN 稀疏注意力宽度/分辨率扫描

RKNN Toolkit2 2.3.2、RK3576 INT8 simulator，K=5 固定 cross 微图。初次 8-channel key 令 `Pad` 落 CPU；改为 16-channel 后四图均为全 NPU。RW 为修复解析器后逐层重新求和的结果。

| token/value 图 | key/value C | 状态 | cycles | RW | placement |
| --- | ---: | ---: | ---: | ---: | --- |
| `45×80` coarse phase | 16/64 | 281 KiB | 0.077M | 6.55 MiB | 全 NPU |
| `90×160` compact | 16/16 | 450 KiB | 0.309M | 12.14 MiB | **全 NPU，保留** |
| `90×160` wider phase | 16/32 | 675 KiB | 0.309M | 16.78 MiB | 全 NPU，整图 RW 余量过小 |
| `180×320` compact | 16/16 | 1.76 MiB | 1.235M | 48.46 MiB | 全 NPU，但部署预算否决 |

`90×160` 16/16 的单独 attention 微图相对 image-only anchor 约增加 3.7% compiler cycles 和 25.4% RW；尚未包含 encoder、polyphase value transform 与融合，必须以整图重新计门槛。微图 simulator 只证明有限输出与 placement，不是真机 latency。

## 17. 新候选：PST-VSR

PST-VSR 把 transformer 与 value reconstruction 明确解耦：

```text
current / cached previous MLVC reconstruction
    ├─ compact key encoder
    │    └─ sparse local cross-attention → offset posterior + confidence
    └─ raw observation polyphase pack
         └─ posterior-weighted splat / count normalization
              └─ small backprojection-inspired residual core → SR
```

冻结原则：

1. attention 只估计“哪个观测坐标可信”，不得先把 value 压成 `45×80` detail；
2. value 在打包前不做 LR 双线性 warp；半像素候选由 phase channel permutation + integer spatial shift 表示；
3. application 可缓存上一帧 MLVC reconstruction，避免让模型递推生成一个会漂移的自由 state；
4. 对静态/低置信区域保留 spatial fallback，posterior/count mask 必须可观测；
5. 首个 deploy prototype 以 `90×160`、16-value 为成本上限，K=13 是否可承受由整图编译决定；若超预算，采用 coarse K=13 coordinate posterior + fine fixed splat，而不是退回 coarse value。

下一门槛依次为：

- RKNN 编译 phase permutation / weighted splat / count normalization 微图；
- 无 oracle motion 的合成 soft-posterior splat，必须保留已知坐标 splat 增益的至少 70%；
- 冻结 spatial backbone 的真实 MLVC 5k-step gate；
- 三种子、QAT、wrong/reverse/reset history 与真机延迟。

在真实 MLVC 质量门槛通过前，`+1.06 dB` 只能称为合成已知运动机制结果，不能称为“划时代性能”。
