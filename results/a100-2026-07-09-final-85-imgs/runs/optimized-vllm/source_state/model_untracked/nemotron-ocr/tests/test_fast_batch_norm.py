# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from nemotron_ocr.inference.models.detector.aspp import ASPP
from nemotron_ocr.inference.models.detector.fast_batch_norm import (
    FusedBatchNormReLU,
    FusedBatchNormReLUAdd,
    FusedResidualBlock,
    fuse_aspp_batch_norm_relu_add,
    fuse_batch_norm_relu,
    fuse_residual_batch_norm_add_relu,
)
from nemotron_ocr.inference.models.detector.regnet import ResBottleneckBlock


def _residual_block(*, projected: bool = False) -> ResBottleneckBlock:
    return ResBottleneckBlock(
        width_in=4,
        width_out=8 if projected else 4,
        stride=2 if projected else 1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        group_width=1,
        bottleneck_multiplier=1.0,
        se_ratio=None,
    )


def test_batch_norm_relu_cpu_fallback_is_exact_and_idempotent():
    original = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU(inplace=True)).eval()
    transformed = copy.deepcopy(original)
    inputs = torch.randn(2, 3, 7, 9)

    assert fuse_batch_norm_relu(transformed) == 1
    assert fuse_batch_norm_relu(transformed) == 0
    assert torch.equal(transformed(inputs), original(inputs))


@pytest.mark.parametrize("projected", [False, True])
def test_residual_cpu_fallback_is_exact(projected):
    original = nn.Sequential(_residual_block(projected=projected)).eval()
    transformed = copy.deepcopy(original)
    inputs = torch.randn(2, 4, 8, 10)

    assert fuse_residual_batch_norm_add_relu(transformed) == 1
    assert torch.equal(transformed(inputs), original(inputs))


def test_aspp_fusion_requires_zero_dropout_eval_residual():
    eligible = ASPP(in_channels=3, num_channels=3, dropout=0.0).eval()
    nonzero_dropout = ASPP(in_channels=3, num_channels=3, dropout=0.5).eval()
    training = ASPP(in_channels=3, num_channels=3, dropout=0.0).train()
    nonresidual = ASPP(in_channels=4, num_channels=3, dropout=0.0).eval()

    assert fuse_aspp_batch_norm_relu_add(eligible) == 1
    assert fuse_aspp_batch_norm_relu_add(nonzero_dropout) == 0
    assert fuse_aspp_batch_norm_relu_add(training) == 0
    assert fuse_aspp_batch_norm_relu_add(nonresidual) == 0


@pytest.mark.parametrize(
    "batch_norm",
    [
        nn.BatchNorm2d(3, affine=False).eval(),
        nn.BatchNorm2d(3, track_running_stats=False).eval(),
        nn.BatchNorm2d(3).train(),
    ],
)
def test_generic_fusion_skips_unsupported_batch_norm(batch_norm):
    module = nn.Sequential(batch_norm, nn.ReLU())

    assert fuse_batch_norm_relu(module) == 0
    assert module[0] is batch_norm


def test_inverse_std_is_nonpersistent_and_refreshes_on_eval():
    module = FusedBatchNormReLU(nn.BatchNorm2d(3).eval())
    original_inverse_std = module.inverse_std.clone()
    assert "inverse_std" not in module.state_dict()

    module.train()
    module(torch.randn(4, 3, 5, 5))
    module.eval()

    assert not torch.equal(module.inverse_std, original_inverse_std)
    assert torch.equal(
        module.inverse_std,
        torch.rsqrt(module.batch_norm.running_var + module.batch_norm.eps),
    )


def test_original_checkpoint_must_load_before_fusion():
    module = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU()).eval()
    original_state = copy.deepcopy(module.state_dict())
    fuse_batch_norm_relu(module)

    with pytest.raises(RuntimeError, match="Missing key"):
        module.load_state_dict(original_state, strict=True)


def test_launch_parameters_are_validated_for_every_wrapper(monkeypatch):
    monkeypatch.setenv("NEMOTRON_OCR_FUSED_BATCH_NORM_BLOCK_SIZE", "3")

    with pytest.raises(ValueError, match="power of two"):
        FusedBatchNormReLU(nn.BatchNorm2d(3).eval())
    with pytest.raises(ValueError, match="power of two"):
        FusedBatchNormReLUAdd(nn.BatchNorm2d(3).eval())
    block = _residual_block().eval()
    final_batch_norm = block.f.c[-1]
    with pytest.raises(ValueError, match="power of two"):
        FusedResidualBlock(block)
    assert block.f.c[-1] is final_batch_norm


def test_num_warps_validation_is_shared(monkeypatch):
    monkeypatch.setenv("NEMOTRON_OCR_FUSED_BATCH_NORM_NUM_WARPS", "16")

    with pytest.raises(ValueError, match="must be one of"):
        FusedBatchNormReLUAdd(nn.BatchNorm2d(3).eval())
