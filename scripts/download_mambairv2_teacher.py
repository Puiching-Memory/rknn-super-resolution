#!/usr/bin/env python3
"""Download MambaIRv2Light ×3 teacher weights for Stage 2 distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.hub import download_url_to_file

RELEASE_URL = (
    "https://github.com/csguoh/MambaIR/releases/download/v1.0/mambairv2_lightSR_x3.pth"
)
DEFAULT_OUTPUT = Path("checkpoints/teacher/mambairv2_lightSR_x3.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing checkpoint can be loaded.",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="HTTP(S) proxy URL (e.g. http://127.0.0.1:7897). "
        "If omitted, uses https_proxy / http_proxy from the environment.",
    )
    return parser.parse_args()


def _apply_proxy(proxy: str | None) -> None:
    import os

    if proxy is None:
        proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if not proxy:
        return
    os.environ["https_proxy"] = proxy
    os.environ["http_proxy"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    print(f"Using proxy: {proxy}")


def verify_checkpoint(path: Path) -> None:
    from rk3588_mobile_sr.models.mambairv2_light import build_mambairv2_light
    from rk3588_mobile_sr.models.teacher_checkpoint import load_teacher_state_dict

    model = build_mambairv2_light(upscale=3)
    state = load_teacher_state_dict(path)
    model.load_state_dict(state, strict=True)
    print(f"OK: {path} ({path.stat().st_size / 1e6:.1f} MB, {len(state)} tensors)")


def main() -> None:
    args = parse_args()
    _apply_proxy(args.proxy)
    if args.verify_only:
        if not args.output.is_file():
            raise SystemExit(f"Missing checkpoint: {args.output}")
        verify_checkpoint(args.output)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file():
        print(f"Already exists: {args.output}")
    else:
        print(f"Downloading {RELEASE_URL}")
        print(f"  -> {args.output}")
        download_url_to_file(RELEASE_URL, str(args.output), progress=True)

    verify_checkpoint(args.output)


if __name__ == "__main__":
    main()
