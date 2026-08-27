#!/usr/bin/env bash
# 把本地最优模型包推送到 Hugging Face。
#
# 用法:
#   HF_TOKEN=hf_xxx ./scripts/hf_upload.sh [repo_name]
#
# 可选环境变量:
#   HF_NAMESPACE    覆盖命名空间(默认取 whoami 的用户名)
#   HF_REPO_NAME    仓库名(等价于第一个位置参数, 默认 phase-rlfn-codec-v1)
#   HF_PACKAGE_DIR  要上传的目录(默认 checkpoints/hf-package/phase-rlfn-codec-v1)
#   HF_PRIVATE=1    创建私有仓库(默认公开)
#
# 需要写权限: 先 `hf auth login`(等价 huggingface_hub.login)或设置 HF_TOKEN。
set -euo pipefail

REPO_NAME="${HF_REPO_NAME:-${1:-phase-rlfn-codec-v1}}"
PKG_DIR="${HF_PACKAGE_DIR:-checkpoints/hf-package/phase-rlfn-codec-v1}"

if [[ ! -d "$PKG_DIR" ]]; then
  echo "错误: 找不到上传目录 $PKG_DIR" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "提示: 未设置 HF_TOKEN。将尝试使用已保存的登录凭据; 若失败请先执行:"
  echo "       huggingface_hub.login()  或   export HF_TOKEN=hf_xxx"
fi

export HF_REPO_NAME="$REPO_NAME" HF_PACKAGE_DIR="$PKG_DIR"

uv run --with 'huggingface-hub==1.23.0' python - <<'PY'
import os, sys
from huggingface_hub import HfApi

pkg = os.environ["HF_PACKAGE_DIR"]
repo_name = os.environ["HF_REPO_NAME"]
private = os.environ.get("HF_PRIVATE", "") in ("1", "true", "yes")
token = os.environ.get("HF_TOKEN") or None

api = HfApi(token=token)
try:
    me = api.whoami(token=token)
except Exception as e:
    print("无法取得 HF 身份, 请先配置 token: %s" % e, file=sys.stderr)
    sys.exit(1)

ns = os.environ.get("HF_NAMESPACE") or me["name"]
repo_id = f"{ns}/{repo_name}"
print("target repo:", repo_id, "| private:", private)

api.create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)
res = api.upload_folder(
    repo_id=repo_id,
    folder_path=pkg,
    repo_type="model",
    commit_message="Upload Phase-RLFN codec-v1 3x SR (MLVC codec-aware, RKNN-ready)",
    token=token,
)
print("upload OK")
print("https://huggingface.co/%s" % repo_id)
PY
