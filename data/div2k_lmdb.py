"""LMDB-backed DIV2K patch cache for fast random-access training."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import lmdb
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from data.div2k_loader import collect_paired_paths

META_KEY = b"__meta__"


@dataclass(frozen=True)
class LMDBMeta:
    patch_size: int
    scale: int
    num_samples: int
    lr_shape: tuple[int, int, int]
    hr_shape: tuple[int, int, int]
    lr_bytes: int
    hr_bytes: int
    offline_augment: bool = False

    @property
    def record_bytes(self) -> int:
        return self.lr_bytes + self.hr_bytes

    def to_json(self) -> str:
        return json.dumps(
            {
                "patch_size": self.patch_size,
                "scale": self.scale,
                "num_samples": self.num_samples,
                "lr_shape": self.lr_shape,
                "hr_shape": self.hr_shape,
                "offline_augment": self.offline_augment,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> LMDBMeta:
        data = json.loads(raw)
        lr_shape = tuple(data["lr_shape"])
        hr_shape = tuple(data["hr_shape"])
        lr_bytes = int(np.prod(lr_shape))
        hr_bytes = int(np.prod(hr_shape))
        return cls(
            patch_size=int(data["patch_size"]),
            scale=int(data["scale"]),
            num_samples=int(data["num_samples"]),
            lr_shape=lr_shape,  # type: ignore[arg-type]
            hr_shape=hr_shape,  # type: ignore[arg-type]
            lr_bytes=lr_bytes,
            hr_bytes=hr_bytes,
            offline_augment=bool(data.get("offline_augment", False)),
        )


def _sample_key(index: int) -> bytes:
    return f"{index:09d}".encode("ascii")


def _crop_aligned_patch(
    lr: torch.Tensor,
    hr: torch.Tensor,
    *,
    patch_size: int,
    scale: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = lr.shape
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        lr = TF.pad(lr, (0, 0, pad_w, pad_h))
        hr = TF.pad(hr, (0, 0, pad_w * scale, pad_h * scale))
        _, h, w = lr.shape

    top = rng.randint(0, h - patch_size)
    left = rng.randint(0, w - patch_size)
    lr_patch = lr[:, top : top + patch_size, left : left + patch_size]
    hr_patch = hr[
        :,
        top * scale : (top + patch_size) * scale,
        left * scale : (left + patch_size) * scale,
    ]
    return lr_patch, hr_patch


def _augment_patch(
    lr: torch.Tensor,
    hr: torch.Tensor,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip/transpose augment; uses global random when rng is None (online train)."""
    coin = rng.random if rng is not None else random.random
    if coin() < 0.5:
        lr = TF.hflip(lr)
        hr = TF.hflip(hr)
    if coin() < 0.5:
        lr = TF.vflip(lr)
        hr = TF.vflip(hr)
    if coin() < 0.5:
        lr = torch.transpose(lr, 1, 2)
        hr = torch.transpose(hr, 1, 2)
    return lr, hr


def pack_patch_pair(lr: torch.Tensor, hr: torch.Tensor) -> bytes:
    lr_u8 = lr.clamp(0.0, 255.0).round().to(torch.uint8).contiguous()
    hr_u8 = hr.clamp(0.0, 255.0).round().to(torch.uint8).contiguous()
    return lr_u8.numpy().tobytes() + hr_u8.numpy().tobytes()


def unpack_patch_pair(data: bytes, meta: LMDBMeta) -> tuple[torch.Tensor, torch.Tensor]:
    lr_arr = np.frombuffer(data[: meta.lr_bytes], dtype=np.uint8).reshape(meta.lr_shape).copy()
    hr_arr = np.frombuffer(
        data[meta.lr_bytes : meta.record_bytes], dtype=np.uint8
    ).reshape(meta.hr_shape).copy()
    return torch.from_numpy(lr_arr).float(), torch.from_numpy(hr_arr).float()


def read_meta(env: lmdb.Environment) -> LMDBMeta:
    with env.begin(write=False) as txn:
        raw = txn.get(META_KEY)
    if raw is None:
        raise ValueError("LMDB is missing metadata key __meta__")
    return LMDBMeta.from_json(raw.decode("utf-8"))


def estimate_map_size(
    num_images: int,
    patches_per_image: int,
    patch_size: int,
    scale: int,
) -> int:
    lr_bytes = 3 * patch_size * patch_size
    hr_bytes = 3 * (patch_size * scale) ** 2
    payload = num_images * patches_per_image * (lr_bytes + hr_bytes)
    # LMDB overhead + metadata headroom.
    return max(int(payload * 1.25) + (1 << 28), 1 << 30)


def build_div2k_lmdb(
    *,
    hr_dir: str,
    lr_dir: str,
    output_dir: str | Path,
    scale: int = 3,
    patch_size: int = 128,
    patches_per_image: int = 32,
    seed: int = 42,
    augment: bool = False,
    subset: str | None = None,
) -> LMDBMeta:
    """Offline-build an LMDB of paired LR/HR uint8 patches (augment at train time by default)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_paths, hr_paths = collect_paired_paths(hr_dir, lr_dir, scale=scale, subset=subset)
    map_size = estimate_map_size(len(lr_paths), patches_per_image, patch_size, scale)

    if (output_dir / "data.mdb").exists():
        raise FileExistsError(
            f"LMDB already exists at {output_dir}. Delete it before rebuilding."
        )

    env = lmdb.open(
        str(output_dir),
        map_size=map_size,
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
    )

    hr_patch = patch_size * scale
    meta = LMDBMeta(
        patch_size=patch_size,
        scale=scale,
        num_samples=len(lr_paths) * patches_per_image,
        lr_shape=(3, patch_size, patch_size),
        hr_shape=(3, hr_patch, hr_patch),
        lr_bytes=3 * patch_size * patch_size,
        hr_bytes=3 * hr_patch * hr_patch,
        offline_augment=augment,
    )

    sample_idx = 0
    txn = env.begin(write=True)
    try:
        for image_idx, (lr_path, hr_path) in enumerate(zip(lr_paths, hr_paths, strict=True)):
            rng = random.Random(seed + image_idx)
            hr = TF.to_tensor(Image.open(hr_path).convert("RGB")) * 255.0
            lr = TF.to_tensor(Image.open(lr_path).convert("RGB")) * 255.0

            for _ in range(patches_per_image):
                lr_patch, hr_patch = _crop_aligned_patch(
                    lr, hr, patch_size=patch_size, scale=scale, rng=rng
                )
                if augment:
                    lr_patch, hr_patch = _augment_patch(lr_patch, hr_patch, rng)
                txn.put(_sample_key(sample_idx), pack_patch_pair(lr_patch, hr_patch))
                sample_idx += 1
                if sample_idx % 512 == 0:
                    txn.commit()
                    txn = env.begin(write=True)

        if sample_idx != meta.num_samples:
            raise RuntimeError(f"Expected {meta.num_samples} samples, wrote {sample_idx}")

        txn.put(META_KEY, meta.to_json().encode("utf-8"))
        txn.commit()
    except Exception:
        txn.abort()
        env.close()
        raise
    finally:
        env.close()

    return meta


class DIV2KLMDBDataset(Dataset):
    """Random-access DIV2K patches stored in LMDB."""

    def __init__(self, lmdb_dir: str | Path, *, augment: bool = True) -> None:
        super().__init__()
        self.lmdb_dir = Path(lmdb_dir)
        self.augment = augment
        self._env: lmdb.Environment | None = None
        self._meta: LMDBMeta | None = None
        self._offline_augment_warned = False

    def _ensure_open(self) -> tuple[lmdb.Environment, LMDBMeta]:
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_dir),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )
            self._meta = read_meta(self._env)
            if (
                self.augment
                and self._meta.offline_augment
                and not self._offline_augment_warned
            ):
                import warnings

                warnings.warn(
                    f"LMDB at {self.lmdb_dir} was built with offline augment; "
                    "rebuild without --offline_augment to avoid double augmentation.",
                    stacklevel=2,
                )
                self._offline_augment_warned = True
        assert self._meta is not None
        return self._env, self._meta

    def __len__(self) -> int:
        _, meta = self._ensure_open()
        return meta.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        env, meta = self._ensure_open()
        key = _sample_key(index)
        with env.begin(write=False) as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(f"Missing LMDB sample {index}")

        lr, hr = unpack_patch_pair(raw, meta)
        if self.augment:
            lr, hr = _augment_patch(lr, hr)
        return lr, hr

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_meta"] = None
        return state


class InfiniteRandomLMDBIterable(IterableDataset):
    """Yield LMDB patches forever via uniform random indices (no epoch boundary)."""

    def __init__(
        self,
        lmdb_dir: str | Path,
        *,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.lmdb_dir = Path(lmdb_dir)
        self.augment = augment
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rank = int(os.environ.get("RANK", "0"))
        rng = random.Random(self.seed + rank * 1_000_003 + worker_id * 10_007)

        dataset = DIV2KLMDBDataset(self.lmdb_dir, augment=self.augment)
        num_samples = len(dataset)
        while True:
            yield dataset[rng.randrange(num_samples)]


class LMDBTrainLoader:
    """Infinite random-sampling iterator for step-based training loops."""

    def __init__(self, dataloader: DataLoader, *, virtual_epoch_steps: int) -> None:
        self.dataloader = dataloader
        self._virtual_epoch_steps = virtual_epoch_steps

    def __len__(self) -> int:
        return self._virtual_epoch_steps

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        yield from self.dataloader


def build_lmdb_train_loader(
    lmdb_dir: str | Path,
    *,
    batch_size: int,
    num_workers: int = 8,
    augment: bool = True,
    distributed: bool = False,
    seed: int = 42,
) -> tuple[LMDBTrainLoader, None]:
    """Build an infinite LMDB training loader with replacement random sampling."""
    del distributed  # each DDP rank draws independently; no DistributedSampler epoch.
    num_samples = len(DIV2KLMDBDataset(lmdb_dir, augment=augment))
    iterable = InfiniteRandomLMDBIterable(lmdb_dir, augment=augment, seed=seed)
    loader = DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    virtual_epoch_steps = max(1, num_samples // batch_size)
    return LMDBTrainLoader(loader, virtual_epoch_steps=virtual_epoch_steps), None


def main() -> None:
    """CLI entry: build DIV2K LMDB patch cache."""
    import argparse

    from data.div2k_loader import collect_paired_paths

    parser = argparse.ArgumentParser(description="Build DIV2K LMDB patch cache for training")
    parser.add_argument("--hr_dir", default="data/DIV2K_train_HR")
    parser.add_argument("--lr_dir", default="data/DIV2K_train_LR_bicubic/X3")
    parser.add_argument("--output_dir", default="data/DIV2K_train_lmdb")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--patches_per_image", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--offline_augment",
        action="store_true",
        help="bake flip/transpose into LMDB at build time (default: augment online during training)",
    )
    args = parser.parse_args()

    lr_paths, _ = collect_paired_paths(args.hr_dir, args.lr_dir, scale=args.scale)
    est_gb = estimate_map_size(
        len(lr_paths), args.patches_per_image, args.patch_size, args.scale
    ) / (1 << 30)
    print(
        f"Building LMDB: {len(lr_paths)} images x {args.patches_per_image} patches, "
        f"patch={args.patch_size}, scale=x{args.scale}, est_size~{est_gb:.1f} GiB"
    )

    meta = build_div2k_lmdb(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        output_dir=args.output_dir,
        scale=args.scale,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        seed=args.seed,
        augment=args.offline_augment,
    )

    print(f"Done: {args.output_dir}")
    print(f"  samples: {meta.num_samples}")
    print(f"  lr shape: {meta.lr_shape}, hr shape: {meta.hr_shape}")
    print(f"  offline_augment: {meta.offline_augment}")
    print(f"\nTrain with:\n  --lmdb_dir {args.output_dir} --no-use_dali")


if __name__ == "__main__":
    main()
