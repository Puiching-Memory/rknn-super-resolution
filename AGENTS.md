# Agent Guidelines

本文件只记录**方法论与不变量**。命令、目录树、训练参数等事实性内容以 [README.md](README.md)、[pyproject.toml](pyproject.toml)、[config/phase_rlfn_sr_x3.yaml](src/rk3588_mobile_sr/config/phase_rlfn_sr_x3.yaml) 为准；改代码时去读源码，不要在这里双向维护副本。

## 信息从哪来

- **怎么跑**：`scripts/*.sh`（运维入口）→ 内部 `uv run` + `torchrun -m ...`；完整 CLI 列表见 `pyproject.toml` `[project.scripts]`。
- **默认配置**：`config/phase_rlfn_sr_x3.yaml`；CLI `--config` / 参数可覆盖。
- **架构与流程**：README；实现细节在 `src/rk3588_mobile_sr/` 对应模块。

## 环境与工具链

- 只用 **`uv`** 管理依赖（`uv sync`）；除非用户明确要求，不用 `pip`。
- `third_party/mlvc` 与 `third_party/vmaf` 是 git 子模块；克隆用 `--recurse-submodules`，不要再 gitignore 后自行 clone。
- `uv sync` 经 hatch 钩子把 libvmaf 装进 `.local/`；验证指标由 `val_metric`（`vmaf`/`psnr`）全阶段统一，没有 `--no_vmaf`。
- 训练栈绑定 **CUDA 13.0 / cu130**；RKNN 工具链与主环境 torch 版本冲突，**隔离在 `.venv-rknn`**，不并入主依赖。
- 校验改动：`uv run pytest`、`uv run ruff check src tests`（配置见 `pyproject.toml`）。

## 代码放哪、怎么改

- 可安装代码一律在 **`src/rk3588_mobile_sr/`**（src layout）。
- **`scripts/` 只放 bash**；Python 工具进包内（如 `dev/`）并注册 `[project.scripts]`，不要复活 `scripts/*.py`。
- 先读周边模块再写：命名、类型、抽象层级与现有文件保持一致；改动范围限于任务所需，不顺手重构无关代码。
- 行为有实质变化时再补测试；不测显然成立的事。
- 不主动改 README / docs、不主动 `git commit`，除非用户要求或对外接口确实变了。

## 架构不变量（违反会出真 bug）

改训练、分布式、数据、可视化相关代码时，默认这些约束仍然成立：

1. **Step-based DDP**：循环在 `train/loop.py`；`train/unified.py` 只组装平台期驱动的
   FP32→QAT 状态机、hook 与配置，不另起一套训练循环。
2. **rank 0 独占段**：日志、SwanLab、checkpoint、验证图像等只能在 collective **完成之后**、经 `distributed/sync.py` `rank0_section` 执行；禁止让非 0 rank 在 broadcast 前空等。
3. **数据契约**：OpenVidHD 以 MLVC `frame_sequences.csv` 为索引，`video_root` 下按文件名解析源 mp4。CPU 只产出 clip 元数据；TorchCodec 在训练 GPU 上按 `start_frame` NVDEC 解码。分辨率与 deploy 对齐（LR 360×640 canvas，HR 1080×1920）。解码与几何变换以 `data/decode.py`、`data/openvid.py`、`data/mlvc_loader.py` 为准。
4. **色彩空间**：默认在 **YUV444** 张量上训练；RGB 仅用于可视化与部分指标，经 `data/yuv_utils.py` 转换。排查色差时先区分 codec/模型误差与 YUV 往返误差（见 `utils/swanlab_logging.py` data preview）。
5. **部署图**：推理前 `switch_to_deploy()`；导出/量化逻辑在 `deploy/`，不与训练循环缠在一起。

## 非显而易见、但值得记住

- `torchrun` 须用 **`-m rk3588_mobile_sr.train.unified`**，不能把 console script 名当文件路径。
- 勿 `export SWANLAB_EXPERIMENT`（SDK 会误解析）；用 `--swanlab_experiment`。`scripts/_common.sh` 已处理 `NO_PROXY` 等环境。
- CUDA `VideoDecoder` 默认走 NVDEC；codec 不受支持或 NVCUVID 缺失时 TorchCodec 会 CPU fallback，不是代码 bug，先查驱动 capabilities。

## SwanLab

写追踪代码或查实验数据时，读 `.cursor/skills/swanlab-skill/`；项目内集成点在 `utils/swanlab_logging.py`。
