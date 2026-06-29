"""GPU-accelerated DIV2K loader using NVIDIA DALI native file readers."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

import nvidia.dali.fn as fn
import nvidia.dali.types as types

from data.div2k_loader import collect_paired_paths

READER_NAME = "Reader"


def _repeat_file_list(files: list[str], times: int) -> list[str]:
    if times <= 1:
        return files
    return [path for path in files for _ in range(times)]


def _shard_file_list(files: list[str], shard_id: int, num_shards: int) -> list[str]:
    return files[shard_id::num_shards]


def _compute_list_repeat(
    num_images: int,
    *,
    batch_size: int,
    num_shards: int,
    samples_per_image: int,
    min_steps_per_epoch: int,
) -> int:
    """Expand the file list so each shard covers at least ``min_steps_per_epoch`` batches."""
    repeat = max(1, samples_per_image)
    while True:
        total = num_images * repeat
        shard_len = (total + num_shards - 1) // num_shards
        if shard_len // batch_size >= min_steps_per_epoch:
            return repeat
        repeat *= 2
        if repeat > 1_048_576:
            raise ValueError(
                f"Cannot reach {min_steps_per_epoch} steps/epoch with {num_images} images"
            )


def _build_train_pipeline(
    *,
    lr_files: list[str],
    hr_files: list[str],
    batch_size: int,
    patch_size: int,
    scale: int,
    device_id: int,
    num_threads: int,
    shard_id: int,
    num_shards: int,
    augment: bool,
    seed: int,
    prefetch_queue_depth: int,
):
    hr_patch = patch_size * scale
    reader_kwargs = {
        "shard_id": shard_id,
        "num_shards": num_shards,
        "random_shuffle": True,
        "shuffle_after_epoch": False,
        "read_ahead": True,
        "prefetch_queue_depth": prefetch_queue_depth,
        "initial_fill": min(512, len(lr_files)),
        "seed": seed,
    }

    @pipeline_def(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
    )
    def div2k_train_pipe():
        lr_encoded, _ = fn.readers.file(files=lr_files, name=READER_NAME, **reader_kwargs)
        hr_encoded, _ = fn.readers.file(files=hr_files, **reader_kwargs)

        crop_pos_x = fn.random.uniform(range=(0.0, 1.0))
        crop_pos_y = fn.random.uniform(range=(0.0, 1.0))
        decode_kwargs = {"device": "mixed", "output_type": types.RGB}
        lr = fn.decoders.image_crop(
            lr_encoded,
            crop=(patch_size, patch_size),
            crop_pos_x=crop_pos_x,
            crop_pos_y=crop_pos_y,
            **decode_kwargs,
        )
        hr = fn.decoders.image_crop(
            hr_encoded,
            crop=(hr_patch, hr_patch),
            crop_pos_x=crop_pos_x,
            crop_pos_y=crop_pos_y,
            **decode_kwargs,
        )

        if augment:
            flip_h = fn.random.coin_flip(probability=0.5)
            flip_v = fn.random.coin_flip(probability=0.5)
            lr = fn.flip(lr, horizontal=flip_h, vertical=flip_v)
            hr = fn.flip(hr, horizontal=flip_h, vertical=flip_v)
            angle = fn.random.uniform(values=[0.0, 90.0, 180.0, 270.0])
            lr = fn.rotate(lr, angle=angle, keep_size=True)
            hr = fn.rotate(hr, angle=angle, keep_size=True)

        lr = fn.crop_mirror_normalize(
            lr,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        )
        hr = fn.crop_mirror_normalize(
            hr,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        )
        return lr, hr

    return div2k_train_pipe


class DIV2KDALILoader:
    """Step-oriented loader backed by DALI file reader with long virtual epochs."""

    def __init__(
        self,
        pipe,
        dali_iter: DALIGenericIterator,
        steps_per_epoch: int,
    ) -> None:
        self.pipe = pipe
        self.dali_iter = dali_iter
        self.steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            for batch in self.dali_iter:
                yield batch[0]["lr"], batch[0]["hr"]


def build_dali_train_loader(
    hr_dir: str,
    lr_dir: str | None,
    *,
    scale: int,
    patch_size: int,
    batch_size: int,
    device_id: int,
    shard_id: int,
    num_shards: int,
    num_threads: int = 8,
    augment: bool = True,
    subset: str | None = None,
    samples_per_image: int = 1,
    min_steps_per_epoch: int = 512,
    prefetch_queue_depth: int = 4,
    seed: int = 42,
) -> DIV2KDALILoader:
    """Build a DALI-backed training loader for one DDP rank."""
    lr_files, hr_files = collect_paired_paths(hr_dir, lr_dir, scale=scale, subset=subset)
    num_images = len(lr_files)
    repeat = _compute_list_repeat(
        num_images,
        batch_size=batch_size,
        num_shards=num_shards,
        samples_per_image=samples_per_image,
        min_steps_per_epoch=min_steps_per_epoch,
    )
    lr_files = _repeat_file_list(lr_files, repeat)
    hr_files = _repeat_file_list(hr_files, repeat)

    if len(lr_files) < num_shards:
        raise ValueError("Not enough training images for the requested shard count")

    pipe_fn = _build_train_pipeline(
        lr_files=lr_files,
        hr_files=hr_files,
        batch_size=batch_size,
        patch_size=patch_size,
        scale=scale,
        device_id=device_id,
        num_threads=num_threads,
        shard_id=shard_id,
        num_shards=num_shards,
        augment=augment,
        seed=seed + shard_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    pipe = pipe_fn()
    pipe.build()

    steps_per_epoch = pipe.epoch_size(READER_NAME) // batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            f"DALI epoch size ({pipe.epoch_size(READER_NAME)}) is smaller than batch_size ({batch_size})"
        )

    dali_iter = DALIGenericIterator(
        [pipe],
        output_map=["lr", "hr"],
        reader_name=READER_NAME,
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
    )
    return DIV2KDALILoader(pipe, dali_iter, steps_per_epoch)
