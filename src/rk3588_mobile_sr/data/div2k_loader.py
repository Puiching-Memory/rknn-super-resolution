"""DIV2K data loader for 3x SISR training."""

import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def collect_paired_paths(
    hr_dir: str,
    lr_dir: str | None,
    scale: int = 3,
    subset: str | None = None,
) -> tuple[list[str], list[str]]:
    """Return aligned LR/HR file path lists."""
    hr_dir_path = Path(hr_dir)
    if not hr_dir_path.exists():
        raise FileNotFoundError(f"HR directory not found: {hr_dir}")

    hr_paths = sorted(hr_dir_path.glob("*.png"))
    if len(hr_paths) == 0:
        hr_paths = sorted(hr_dir_path.glob("*.jpg"))
    if subset is not None:
        indices = [int(x) for x in subset.replace(",", " ").split()]
        hr_paths = [hr_paths[i - 1] for i in indices if 0 < i <= len(hr_paths)]

    if lr_dir is None:
        raise ValueError("lr_dir is required for paired DIV2K loading")

    lr_dir_path = Path(lr_dir)
    lr_files = sorted(lr_dir_path.glob(f"*x{scale}.png"))
    if len(lr_files) == 0:
        lr_files = sorted(lr_dir_path.glob("*.png"))
        if len(lr_files) == 0:
            lr_files = sorted(lr_dir_path.glob("*.jpg"))
    lr_map = {p.stem.split("x")[0]: p for p in lr_files}

    lr_paths: list[str] = []
    hr_matched: list[str] = []
    for hr_path in hr_paths:
        key = hr_path.stem
        if key not in lr_map:
            continue
        lr_paths.append(str(lr_map[key]))
        hr_matched.append(str(hr_path))

    if len(lr_paths) == 0:
        raise FileNotFoundError(f"No paired LR/HR images found under {hr_dir} and {lr_dir}")

    return lr_paths, hr_matched


class DIV2KDataset(Dataset):
    """DIV2K dataset returning paired LR/HR patches."""

    def __init__(
        self,
        hr_dir: str,
        lr_dir: str | None = None,
        scale: int = 3,
        patch_size: int = 128,
        augment: bool = True,
        subset: str | None = None,
    ):
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.hr_patch_size = patch_size * scale
        self.augment = augment

        if lr_dir is None:
            raise ValueError("lr_dir is required")

        lr_paths, hr_paths = collect_paired_paths(hr_dir, lr_dir, scale=scale, subset=subset)
        self.lr_paths = [Path(p) for p in lr_paths]
        self.hr_paths = [Path(p) for p in hr_paths]

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr = Image.open(self.hr_paths[idx]).convert("RGB")
        lr = Image.open(self.lr_paths[idx]).convert("RGB")

        hr = TF.to_tensor(hr) * 255.0
        lr = TF.to_tensor(lr) * 255.0

        # Random crop aligned LR/HR patch
        _, h, w = lr.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            lr = TF.pad(lr, (0, 0, pad_w, pad_h))
            hr = TF.pad(hr, (0, 0, pad_w * self.scale, pad_h * self.scale))
            _, h, w = lr.shape

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        lr_patch = lr[:, top : top + self.patch_size, left : left + self.patch_size]
        hr_patch = hr[
            :,
            top * self.scale : (top + self.patch_size) * self.scale,
            left * self.scale : (left + self.patch_size) * self.scale,
        ]

        if self.augment:
            lr_patch, hr_patch = self._augment(lr_patch, hr_patch)

        return lr_patch, hr_patch

    @staticmethod
    def _augment(lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)
        if random.random() < 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)
        if random.random() < 0.5:
            lr = torch.transpose(lr, 1, 2)
            hr = torch.transpose(hr, 1, 2)
        return lr, hr


def build_dataloader(
    hr_dir: str,
    lr_dir: str | None,
    scale: int,
    patch_size: int,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = True,
    subset: str | None = None,
    distributed: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = DIV2KDataset(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        scale=scale,
        patch_size=patch_size,
        augment=augment,
        subset=subset,
    )
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    ), sampler
