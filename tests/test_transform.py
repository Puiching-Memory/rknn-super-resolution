"""Augmentation operator tests."""

from __future__ import annotations

import kornia.augmentation as K
import torch

from rk3588_mobile_sr.data.augment import (
    AugmentConfig,
    augment_config_for_canvas,
    batch_augment_pair,
    batch_random_crop_pair,
    build_paired_augment,
)
from rk3588_mobile_sr.data.train_loader import apply_canvas_batch_transform


def test_batch_random_crop_pair():
    lr = torch.randn(2, 3, 360, 640)
    hr = torch.randn(2, 3, 1080, 1920)
    out_lr, out_hr = batch_random_crop_pair(
        lr, hr, lr_crop_h=128, lr_crop_w=128, scale=3
    )
    assert out_lr.shape == (2, 3, 128, 128)
    assert out_hr.shape == (2, 3, 384, 384)


def test_kornia_rotate90_square_patch():
    aug = K.AugmentationSequential(
        K.RandomRotation90(times=(1, 1), p=1.0),
        data_keys=["input", "input"],
    )
    lr = torch.randn(1, 3, 128, 128)
    hr = torch.randn(1, 3, 384, 384)
    lr_r, hr_r = aug(lr, hr)
    assert lr_r.shape == (1, 3, 128, 128)
    assert hr_r.shape == (1, 3, 384, 384)


def test_kornia_rotate90_rectangular_keepdim():
    config = AugmentConfig(rot90=True, rot90_resize_back=True, rot180=False)
    aug = build_paired_augment(config)
    lr = torch.randn(1, 3, 360, 640)
    hr = torch.randn(1, 3, 1080, 1920)
    lr_r, hr_r = aug(lr, hr)
    assert lr_r.shape == (1, 3, 360, 640)
    assert hr_r.shape == (1, 3, 1080, 1920)


def test_batch_augment_rectangular_rot90_disabled_by_default():
    torch.manual_seed(0)
    lr = torch.randn(4, 3, 360, 640)
    hr = torch.randn(4, 3, 1080, 1920)
    out_lr, out_hr = batch_augment_pair(
        lr,
        hr,
        config=AugmentConfig(rot90=False),
        lr_canvas=(360, 640),
        hr_canvas=(1080, 1920),
    )
    assert out_lr.shape == (4, 3, 360, 640)
    assert out_hr.shape == (4, 3, 1080, 1920)


def test_kornia_flip_horizontal():
    aug = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0),
        data_keys=["input", "input"],
    )
    lr = torch.tensor([[[[0.0, 1.0]]]]).expand(1, 3, 1, 2).clone()
    hr = lr.clone()
    out_lr, _ = aug(lr, hr)
    assert out_lr[0, 0, 0, 0] == 1.0
    assert out_lr[0, 0, 0, 1] == 0.0


def test_paired_augment_same_params():
    """LR and HR must receive identical geometric transforms."""
    aug = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0),
        data_keys=["input", "input"],
    )
    lr = torch.zeros(1, 1, 8, 8)
    lr[0, 0, 0, 0] = 1.0
    hr = torch.zeros(1, 1, 24, 24)
    hr[0, 0, 0, 0] = 1.0
    out_lr, out_hr = aug(lr, hr)
    assert out_lr[0, 0, 0, -1] == 1.0
    assert out_hr[0, 0, 0, -1] == 1.0


def test_lr_degrade_leaves_hr_unchanged():
    config = AugmentConfig(
        flip_h=False,
        flip_v=False,
        rot180=False,
        lr_decode_noise=True,
        decode_noise_p=1.0,
        decode_noise_std=10.0,
    )
    lr = torch.full((2, 3, 32, 32), 128.0)
    hr = torch.full((2, 3, 96, 96), 200.0)
    out_lr, out_hr = batch_augment_pair(
        lr,
        hr,
        config=config,
        lr_canvas=(32, 32),
        hr_canvas=(96, 96),
    )
    assert not torch.allclose(out_lr, lr)
    assert torch.allclose(out_hr, hr)


def test_augment_config_for_canvas_lr_flags():
    cfg = augment_config_for_canvas(
        augment_lr_decode_noise=True,
        augment_lr_jpeg=True,
        patch_size=128,
    )
    assert cfg.lr_decode_noise is True
    assert cfg.lr_jpeg is True
    assert cfg.rot90 is True


def test_canvas_resize_contract():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = torch.randint(0, 255, (2, 3, 48, 64), dtype=torch.uint8)
    hr = torch.randint(0, 255, (2, 3, 200, 300), dtype=torch.uint8)
    out_lr, out_hr = apply_canvas_batch_transform(
        lr,
        hr,
        lr_size=(360, 640),
        hr_size=(1080, 1920),
        colorspace="rgb",
        augment=False,
        device=device,
    )
    assert out_lr.shape == (2, 3, 360, 640)
    assert out_hr.shape == (2, 3, 1080, 1920)


def test_canvas_patch_crop_contract():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = torch.randint(0, 255, (2, 3, 360, 640), dtype=torch.uint8)
    hr = torch.randint(0, 255, (2, 3, 1080, 1920), dtype=torch.uint8)
    out_lr, out_hr = apply_canvas_batch_transform(
        lr,
        hr,
        lr_size=(360, 640),
        hr_size=(1080, 1920),
        colorspace="rgb",
        augment=False,
        device=device,
        patch_size=128,
        scale=3,
    )
    assert out_lr.shape == (2, 3, 128, 128)
    assert out_hr.shape == (2, 3, 384, 384)
