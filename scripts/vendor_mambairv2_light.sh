#!/usr/bin/env bash
# Re-vendor MambaIRv2Light from upstream MambaIR (requires network; use proxy if needed).
#
# Example (adjust host/port to your proxy):
#   export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
#   ./scripts/vendor_mambairv2_light.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="${TMPDIR:-/tmp}/MambaIR-vendor-$$"
UPSTREAM="https://github.com/csguoh/MambaIR.git"
SRC_REL="basicsr/archs/mambairv2light_arch.py"
DST="${ROOT}/src/rk3588_mobile_sr/models/mambairv2_light.py"

if [[ -n "${https_proxy:-}" || -n "${HTTPS_PROXY:-}" ]]; then
  echo "Using proxy: ${https_proxy:-${HTTPS_PROXY:-}}"
else
  echo "Tip: GitHub clone may need a proxy, e.g."
  echo "  export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897"
fi

git clone --depth 1 "${UPSTREAM}" "${TMP_DIR}"
test -f "${TMP_DIR}/${SRC_REL}"

python3 - "${TMP_DIR}/${SRC_REL}" "${DST}" <<'PY'
import pathlib
import sys

src, dst = map(pathlib.Path, sys.argv[1:3])
text = src.read_text(encoding="utf-8")
header = '''# Vendored from https://github.com/csguoh/MambaIR (basicsr/archs/mambairv2light_arch.py).
# MIT License — see MambaIR repository for upstream terms.

from __future__ import annotations

'''
text = text.replace(
    "import math\nimport numpy as np\nimport torch\nimport torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from basicsr.archs.arch_util import to_2tuple, trunc_normal_\n"
    "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref\n"
    "from basicsr.utils.registry import ARCH_REGISTRY\n"
    "from einops import rearrange, repeat\n\n\n",
    "import math\n\nimport numpy as np\nimport torch\nimport torch.nn as nn\n"
    "import torch.nn.functional as F\nfrom einops import rearrange, repeat\n"
    "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn\n\n"
    "from rk3588_mobile_sr.models.arch_util import to_2tuple, trunc_normal_\n\n\n",
)
text = text.replace("@ARCH_REGISTRY.register()\n", "")
if "def build_mambairv2_light" not in text:
    text = text.rstrip() + '''

def build_mambairv2_light(*, upscale: int = 3) -> MambaIRv2Light:
    """MambaIRv2Light config matching official lightSR ×3 test yml."""
    return MambaIRv2Light(
        upscale=upscale,
        in_chans=3,
        img_size=64,
        img_range=1.0,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=1.0,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
    )
'''
# Drop upstream __main__ smoke test if present.
marker = "\nif __name__ == '__main__':"
if marker in text:
    text = text[: text.index(marker)].rstrip() + "\n"
dst.write_text(header + text.lstrip(), encoding="utf-8")
print(f"Wrote {dst}")
PY

rm -rf "${TMP_DIR}"
echo "Done. Run: uv run ruff check src/rk3588_mobile_sr/models/mambairv2_light.py"
