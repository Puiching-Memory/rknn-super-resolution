"""Thin command dispatcher for training, evaluation, export, and deployment."""

from __future__ import annotations

import importlib
import sys

_COMMANDS = {
    "train": ("rknn_super_resolution.train.unified", "main"),
    "eval-mlvc-checkpoint": ("rknn_super_resolution.eval.mlvc_checkpoint", "main"),
    "export-onnx": ("rknn_super_resolution.deploy.onnx", "main"),
    "convert-rknn": ("rknn_super_resolution.deploy.rknn", "main"),
}


def _print_help() -> None:
    commands = "\n".join(f"  {name}" for name in _COMMANDS)
    print(
        "Rockchip RKNN Phase-RLFN 3x video super-resolution\n\n"
        "Usage: rknn-super-resolution COMMAND [ARGS...]\n\n"
        f"Commands:\n{commands}"
    )


def main() -> None:
    """Dispatch to the argparse CLI owned by each subsystem."""
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_help()
        return
    command = sys.argv[1]
    target = _COMMANDS.get(command)
    if target is None:
        choices = ", ".join(_COMMANDS)
        raise SystemExit(f"unknown command {command!r}; expected one of: {choices}")
    module_name, function_name = target
    sys.argv = [command, *sys.argv[2:]]
    getattr(importlib.import_module(module_name), function_name)()


if __name__ == "__main__":
    main()
