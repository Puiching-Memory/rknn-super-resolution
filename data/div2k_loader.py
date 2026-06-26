"""DIV2K data loader for 3x SISR training."""

import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler


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

        hr_dir = Path(hr_dir)
        if not hr_dir.exists():
            raise FileNotFoundError(f"HR directory not found: {hr_dir}")

        self.hr_paths = sorted(hr_dir.glob("*.png"))
        if len(self.hr_paths) == 0:
            self.hr_paths = sorted(hr_dir.glob("*.jpg"))
        if subset is not None:
            indices = [int(x) for x in subset.replace(",", " ").split()]
            self.hr_paths = [self.hr_paths[i - 1] for i in indices if 0 < i <= len(self.hr_paths)]

        self.lr_paths = None
        if lr_dir is not None:
            lr_dir = Path(lr_dir)
            lr_files = sorted(lr_dir.glob(f"*x{scale}.png"))
            if len(lr_files) == 0:
                lr_files = sorted(lr_dir.glob("*.png"))
                if len(lr_files) == 0:
                    lr_files = sorted(lr_dir.glob("*.jpg"))
            self.lr_paths = {p.stem.split("x")[0]: p for p in lr_files}

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr_path = self.hr_paths[idx]
        hr = Image.open(hr_path).convert("RGB")

        if self.lr_paths is not None:
            key = hr_path.stem
            lr = Image.open(self.lr_paths[key]).convert("RGB")
        else:
            w, h = hr.size
            lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)

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
        shuffle=(augment and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    ), sampler
