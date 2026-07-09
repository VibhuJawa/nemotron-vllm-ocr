# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import nemotron_ocr.inference.models.detector.aspp as aspp_module
import nemotron_ocr.inference.models.detector.fots_detector as fots_module
from nemotron_ocr.inference.models.detector.aspp import ASPP, _aspp_concat
from nemotron_ocr.inference.models.detector.fots_detector import (
    _upsample_concat_nearest,
    merge,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("batch", "input_channels", "side_channels", "height", "width"),
    [(1, 1, 2, 2, 3), (2, 3, 1, 3, 2), (1, 2, 3, 1, 5)],
)
def test_upsample_concat_cpu_fallback_is_exact(
    dtype, batch, input_channels, side_channels, height, width
):
    input_tensor = torch.arange(
        batch * input_channels * height * width, dtype=torch.float32
    ).reshape(batch, input_channels, height, width)
    input_tensor = input_tensor.to(dtype)
    side_tensor = torch.arange(
        batch * side_channels * height * 2 * width * 2, dtype=torch.float32
    ).reshape(batch, side_channels, height * 2, width * 2)
    side_tensor = side_tensor.to(dtype).add(100)

    expected = torch.cat(
        (F.interpolate(input_tensor, scale_factor=2, mode="nearest"), side_tensor),
        dim=1,
    )
    actual = _upsample_concat_nearest(input_tensor, side_tensor)

    assert actual.dtype == dtype
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("batch", "channels", "height", "width"),
    [(1, 2, 3, 5), (2, 1, 1, 4), (2, 3, 2, 2)],
)
def test_aspp_concat_cpu_fallback_is_exact(dtype, batch, channels, height, width):
    elements = batch * channels * height * width
    base = torch.arange(elements, dtype=torch.float32).reshape(
        batch, channels, height, width
    )
    branches = [(base + branch * 100).to(dtype) for branch in range(7)]
    pooled = torch.arange(batch * channels, dtype=torch.float32).reshape(
        batch, channels, 1, 1
    )
    pooled = pooled.to(dtype).add(1000)

    expected_pooled = pooled.expand(-1, -1, height, width)
    expected = torch.cat([*branches, expected_pooled], dim=1)
    actual = _aspp_concat(branches, pooled)

    assert actual.dtype == dtype
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    assert torch.equal(actual[:, 7 * channels :], expected_pooled)


class _FakeCudaTensor:
    def __init__(self, shape, *, contiguous=True):
        self.shape = shape
        self.is_cuda = True
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


@pytest.mark.parametrize(
    ("input_tensor", "side_tensor", "message"),
    [
        (
            _FakeCudaTensor((1, 2, 2, 3)),
            _FakeCudaTensor((2, 4, 4, 6)),
            "fused upsample-concat requires an exact 2x side tensor",
        ),
        (
            _FakeCudaTensor((1, 2, 2, 3)),
            _FakeCudaTensor((1, 4, 5, 6)),
            "fused upsample-concat requires an exact 2x side tensor",
        ),
        (
            _FakeCudaTensor((1, 2, 2, 3)),
            _FakeCudaTensor((1, 4, 4, 5)),
            "fused upsample-concat requires an exact 2x side tensor",
        ),
    ],
)
def test_upsample_concat_validation_without_cuda(
    monkeypatch, input_tensor, side_tensor, message
):
    monkeypatch.setattr(fots_module, "triton", object())

    with pytest.raises(ValueError, match=message):
        _upsample_concat_nearest(input_tensor, side_tensor)


def test_upsample_concat_noncontiguous_cuda_falls_back(monkeypatch):
    input_tensor = _FakeCudaTensor((1, 2, 2, 3), contiguous=False)
    side_tensor = _FakeCudaTensor((1, 4, 4, 6))
    sentinel = object()
    monkeypatch.setattr(fots_module, "triton", object())
    monkeypatch.setattr(fots_module.F, "interpolate", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(
        fots_module.torch,
        "cat",
        lambda tensors, dim: (tensors, dim),
    )

    tensors, dim = _upsample_concat_nearest(input_tensor, side_tensor)

    assert tensors == (sentinel, side_tensor)
    assert dim == 1


def test_aspp_concat_requires_seven_branches_without_cuda(monkeypatch):
    monkeypatch.setattr(aspp_module, "triton", object())
    branch = _FakeCudaTensor((1, 2, 3, 4))

    with pytest.raises(
        ValueError, match="fused ASPP concat expects seven spatial branches"
    ):
        _aspp_concat([branch] * 6, _FakeCudaTensor((1, 2, 1, 1)))


def test_aspp_concat_noncontiguous_cuda_falls_back(monkeypatch):
    monkeypatch.setattr(aspp_module, "triton", object())
    branches = [_FakeCudaTensor((1, 2, 3, 4)) for _ in range(7)]
    branches[3] = _FakeCudaTensor((1, 2, 3, 4), contiguous=False)
    pooled = _FakeCudaTensor((1, 2, 1, 1))
    sentinel = object()
    monkeypatch.setattr(pooled, "expand", lambda *args: sentinel, raising=False)
    monkeypatch.setattr(
        aspp_module.torch,
        "cat",
        lambda tensors, dim: (tensors, dim),
    )

    tensors, dim = _aspp_concat(branches, pooled)

    assert tensors == [*branches, sentinel]
    assert dim == 1


def test_aspp_concat_validates_pooled_shape_without_cuda(monkeypatch):
    monkeypatch.setattr(aspp_module, "triton", object())
    branches = [_FakeCudaTensor((2, 3, 4, 5)) for _ in range(7)]

    with pytest.raises(
        ValueError, match="fused ASPP pooled branch has an unexpected shape"
    ):
        _aspp_concat(branches, _FakeCudaTensor((2, 3, 4, 5)))


def test_exact_copy_helpers_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NEMOTRON_OCR_FUSED_UPSAMPLE_CONCAT", raising=False)
    monkeypatch.delenv("NEMOTRON_OCR_FUSED_ASPP_CONCAT", raising=False)

    assert merge([4])._fused_upsample_concat is False
    assert ASPP(in_channels=1, num_channels=1)._fused_concat is False
