# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional inference-only fused BatchNorm detector kernels.

Fusion is a serve-time transformation: load the original checkpoint first,
put the detector in eval mode, and only then call the helpers in this module.
The wrappers intentionally change module structure and therefore do not accept
the original checkpoint key layout after fusion.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


_VALID_NUM_WARPS = (1, 2, 4, 8)


def _batch_norm_launch_parameters() -> tuple[int, int]:
    """Read and validate the launch configuration shared by every wrapper."""
    try:
        block_size = int(
            os.environ.get("NEMOTRON_OCR_FUSED_BATCH_NORM_BLOCK_SIZE", "2048")
        )
        num_warps = int(os.environ.get("NEMOTRON_OCR_FUSED_BATCH_NORM_NUM_WARPS", "8"))
    except ValueError as exc:
        raise ValueError("fused BatchNorm launch parameters must be integers") from exc
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("fused BatchNorm block size must be a power of two")
    if num_warps not in _VALID_NUM_WARPS:
        raise ValueError(f"fused BatchNorm num warps must be one of {_VALID_NUM_WARPS}")
    return block_size, num_warps


def _is_standard_eval_batch_norm(module: nn.Module) -> bool:
    """Return whether a BatchNorm has every tensor required by the kernels."""
    return (
        isinstance(module, nn.BatchNorm2d)
        and not module.training
        and module.affine
        and module.track_running_stats
        and module.weight is not None
        and module.bias is not None
        and module.running_mean is not None
        and module.running_var is not None
    )


def _require_standard_eval_batch_norm(batch_norm: nn.Module) -> nn.BatchNorm2d:
    if not _is_standard_eval_batch_norm(batch_norm):
        raise ValueError(
            "fused BatchNorm requires an affine, running-stat-tracked "
            "BatchNorm2d already in eval mode"
        )
    return batch_norm


class _FusedBatchNormModule(nn.Module):
    """Refresh nonpersistent derived buffers whenever a wrapper returns to eval."""

    def _inverse_std_sources(self) -> tuple[tuple[str, nn.BatchNorm2d], ...]:
        return ()

    def _register_inverse_std(self, name: str, batch_norm: nn.BatchNorm2d) -> None:
        self.register_buffer(
            name,
            torch.rsqrt(batch_norm.running_var + batch_norm.eps),
            persistent=False,
        )

    @torch.no_grad()
    def _refresh_inverse_std(self) -> None:
        for name, batch_norm in self._inverse_std_sources():
            refreshed = torch.rsqrt(batch_norm.running_var + batch_norm.eps)
            current = getattr(self, name)
            if (
                current.shape == refreshed.shape
                and current.dtype == refreshed.dtype
                and current.device == refreshed.device
            ):
                current.copy_(refreshed)
            else:
                setattr(self, name, refreshed)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self._refresh_inverse_std()
        return self


@triton.jit
def _batch_norm_relu_kernel(
    input_ptr,
    mean_ptr,
    inverse_std_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    spatial_size: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    spatial_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    batch_channel = tl.program_id(1)
    channel = batch_channel % channels
    offsets = batch_channel * spatial_size + spatial_offsets
    valid = spatial_offsets < spatial_size
    values = tl.load(input_ptr + offsets, mask=valid).to(tl.float32)
    # These are scalar values shared by the whole spatial tile.  A block mask
    # is invalid for a scalar pointer in recent Triton releases and would also
    # imply redundant vector loads.
    mean = tl.load(mean_ptr + channel)
    inverse_std = tl.load(inverse_std_ptr + channel)
    weight = tl.load(weight_ptr + channel)
    bias = tl.load(bias_ptr + channel)
    # Match cuDNN's fp32 inference arithmetic exactly: apply the affine weight
    # to the centered value first, then use a round-to-nearest fused
    # multiply-add for inverse standard deviation and bias.  Reassociating
    # these products changes rare fp16 rounding boundaries and compounds
    # through the detector.
    normalized = libdevice.fma_rn((values - mean) * weight, inverse_std, bias)
    rounded = normalized.to(tl.float16)
    tl.store(output_ptr + offsets, tl.maximum(rounded, 0.0), mask=valid)


@triton.jit
def _batch_norm_add_relu_kernel(
    main_ptr,
    residual_ptr,
    mean_ptr,
    inverse_std_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    spatial_size: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    spatial_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    batch_channel = tl.program_id(1)
    channel = batch_channel % channels
    offsets = batch_channel * spatial_size + spatial_offsets
    valid = spatial_offsets < spatial_size

    main = tl.load(main_ptr + offsets, mask=valid).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=valid).to(tl.float32)
    mean = tl.load(mean_ptr + channel)
    inverse_std = tl.load(inverse_std_ptr + channel)
    weight = tl.load(weight_ptr + channel)
    bias = tl.load(bias_ptr + channel)

    normalized = libdevice.fma_rn((main - mean) * weight, inverse_std, bias).to(
        tl.float16
    )
    # The eager graph rounds BatchNorm to fp16, then performs an fp16
    # residual add before ReLU.  Preserve both rounding boundaries.
    summed = (normalized.to(tl.float32) + residual).to(tl.float16)
    tl.store(output_ptr + offsets, tl.maximum(summed, 0.0), mask=valid)


@triton.jit
def _batch_norm_relu_add_kernel(
    input_ptr,
    residual_ptr,
    mean_ptr,
    inverse_std_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    spatial_size: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    spatial_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    batch_channel = tl.program_id(1)
    channel = batch_channel % channels
    offsets = batch_channel * spatial_size + spatial_offsets
    valid = spatial_offsets < spatial_size
    values = tl.load(input_ptr + offsets, mask=valid).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=valid).to(tl.float32)
    normalized = libdevice.fma_rn(
        (values - tl.load(mean_ptr + channel)) * tl.load(weight_ptr + channel),
        tl.load(inverse_std_ptr + channel),
        tl.load(bias_ptr + channel),
    ).to(tl.float16)
    activated = tl.maximum(normalized, 0.0)
    # ASPP applies its residual add after the fp16 ReLU.
    summed = (activated.to(tl.float32) + residual).to(tl.float16)
    tl.store(output_ptr + offsets, summed, mask=valid)


@triton.jit
def _dual_batch_norm_add_relu_kernel(
    main_ptr,
    residual_ptr,
    main_mean_ptr,
    main_inverse_std_ptr,
    main_weight_ptr,
    main_bias_ptr,
    residual_mean_ptr,
    residual_inverse_std_ptr,
    residual_weight_ptr,
    residual_bias_ptr,
    output_ptr,
    spatial_size: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    spatial_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    batch_channel = tl.program_id(1)
    channel = batch_channel % channels
    offsets = batch_channel * spatial_size + spatial_offsets
    valid = spatial_offsets < spatial_size

    main = tl.load(main_ptr + offsets, mask=valid).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=valid).to(tl.float32)

    main_normalized = libdevice.fma_rn(
        (main - tl.load(main_mean_ptr + channel)) * tl.load(main_weight_ptr + channel),
        tl.load(main_inverse_std_ptr + channel),
        tl.load(main_bias_ptr + channel),
    ).to(tl.float16)
    residual_normalized = libdevice.fma_rn(
        (residual - tl.load(residual_mean_ptr + channel))
        * tl.load(residual_weight_ptr + channel),
        tl.load(residual_inverse_std_ptr + channel),
        tl.load(residual_bias_ptr + channel),
    ).to(tl.float16)

    summed = (main_normalized.to(tl.float32) + residual_normalized.to(tl.float32)).to(
        tl.float16
    )
    tl.store(output_ptr + offsets, tl.maximum(summed, 0.0), mask=valid)


class FusedBatchNormReLU(_FusedBatchNormModule):
    """Evaluate an existing BatchNorm2d and ReLU in one Triton pass."""

    def __init__(self, batch_norm: nn.BatchNorm2d):
        super().__init__()
        self.batch_norm = _require_standard_eval_batch_norm(batch_norm)
        self._register_inverse_std("inverse_std", self.batch_norm)
        self.block_size, self.num_warps = _batch_norm_launch_parameters()
        # Replacement happens after the detector has been put in eval mode.
        # New ``nn.Module`` instances otherwise default to training mode and
        # silently route every inference call through the fallback below.
        self.train(self.batch_norm.training)

    def _inverse_std_sources(self):
        return (("inverse_std", self.batch_norm),)

    def forward(self, input_tensor):
        if (
            not input_tensor.is_cuda
            or input_tensor.dtype != torch.float16
            or not input_tensor.is_contiguous()
            or self.batch_norm.training
        ):
            return F.relu(self.batch_norm(input_tensor), inplace=True)

        _, channels, height, width = input_tensor.shape
        output = torch.empty_like(input_tensor)
        grid = (
            triton.cdiv(height * width, self.block_size),
            input_tensor.numel() // (height * width),
        )
        _batch_norm_relu_kernel[grid](
            input_tensor,
            self.batch_norm.running_mean,
            self.inverse_std,
            self.batch_norm.weight,
            self.batch_norm.bias,
            output,
            spatial_size=height * width,
            channels=channels,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
        )
        return output


class FusedBatchNormReLUAdd(_FusedBatchNormModule):
    """Fuse BatchNorm, ReLU, then a residual add in exact eager order."""

    def __init__(self, batch_norm: nn.BatchNorm2d):
        super().__init__()
        self.batch_norm = _require_standard_eval_batch_norm(batch_norm)
        self._register_inverse_std("inverse_std", self.batch_norm)
        self.block_size, self.num_warps = _batch_norm_launch_parameters()
        self.train(self.batch_norm.training)

    def _inverse_std_sources(self):
        return (("inverse_std", self.batch_norm),)

    def forward(self, input_tensor, residual):
        if (
            not input_tensor.is_cuda
            or not residual.is_cuda
            or input_tensor.dtype != torch.float16
            or residual.dtype != torch.float16
            or not input_tensor.is_contiguous()
            or not residual.is_contiguous()
            or input_tensor.shape != residual.shape
            or self.batch_norm.training
        ):
            return residual + F.relu(self.batch_norm(input_tensor), inplace=True)

        _, channels, height, width = input_tensor.shape
        output = torch.empty_like(input_tensor)
        grid = (
            triton.cdiv(height * width, self.block_size),
            input_tensor.numel() // (height * width),
        )
        _batch_norm_relu_add_kernel[grid](
            input_tensor,
            residual,
            self.batch_norm.running_mean,
            self.inverse_std,
            self.batch_norm.weight,
            self.batch_norm.bias,
            output,
            spatial_size=height * width,
            channels=channels,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
        )
        return output


class FusedResidualBlock(_FusedBatchNormModule):
    """Fuse final BatchNorm(s), residual add, and ReLU in a RegNet block."""

    def __init__(self, block: nn.Module):
        super().__init__()
        if block.training:
            raise ValueError("fused residual blocks require eval mode")
        main_batch_norm = _require_standard_eval_batch_norm(block.f.c[-1])
        projection_batch_norm = None
        if block.proj is not None:
            projection_batch_norm = _require_standard_eval_batch_norm(block.proj[-1])
        block_size, num_warps = _batch_norm_launch_parameters()

        self.transform = block.f
        self.projection = block.proj
        self.main_batch_norm = main_batch_norm
        self.transform.c[-1] = nn.Identity()
        self._register_inverse_std("main_inverse_std", self.main_batch_norm)

        self.projection_batch_norm = projection_batch_norm
        if self.projection is not None:
            self.projection[-1] = nn.Identity()
            self._register_inverse_std(
                "projection_inverse_std",
                self.projection_batch_norm,
            )

        self.block_size, self.num_warps = block_size, num_warps
        self.train(block.training)

    def _inverse_std_sources(self):
        sources = [("main_inverse_std", self.main_batch_norm)]
        if self.projection_batch_norm is not None:
            sources.append(("projection_inverse_std", self.projection_batch_norm))
        return tuple(sources)

    def _can_fuse(self, main, residual):
        return (
            main.is_cuda
            and residual.is_cuda
            and main.dtype == torch.float16
            and residual.dtype == torch.float16
            and main.is_contiguous()
            and residual.is_contiguous()
            and main.shape == residual.shape
            and not self.main_batch_norm.training
            and (
                self.projection_batch_norm is None
                or not self.projection_batch_norm.training
            )
        )

    def forward(self, input_tensor):
        if self.projection is None:
            residual = input_tensor
            main = self.transform(input_tensor)
        else:
            # Preserve the original block's left-to-right evaluation order.
            residual = self.projection(input_tensor)
            main = self.transform(input_tensor)

        if not self._can_fuse(main, residual):
            main = self.main_batch_norm(main)
            if self.projection_batch_norm is not None:
                residual = self.projection_batch_norm(residual)
            return F.relu(main + residual, inplace=True)

        _, channels, height, width = main.shape
        output = torch.empty_like(main)
        grid = (
            triton.cdiv(height * width, self.block_size),
            main.numel() // (height * width),
        )
        if self.projection_batch_norm is None:
            _batch_norm_add_relu_kernel[grid](
                main,
                residual,
                self.main_batch_norm.running_mean,
                self.main_inverse_std,
                self.main_batch_norm.weight,
                self.main_batch_norm.bias,
                output,
                spatial_size=height * width,
                channels=channels,
                BLOCK_SIZE=self.block_size,
                num_warps=self.num_warps,
            )
        else:
            _dual_batch_norm_add_relu_kernel[grid](
                main,
                residual,
                self.main_batch_norm.running_mean,
                self.main_inverse_std,
                self.main_batch_norm.weight,
                self.main_batch_norm.bias,
                self.projection_batch_norm.running_mean,
                self.projection_inverse_std,
                self.projection_batch_norm.weight,
                self.projection_batch_norm.bias,
                output,
                spatial_size=height * width,
                channels=channels,
                BLOCK_SIZE=self.block_size,
                num_warps=self.num_warps,
            )
        return output


def fuse_batch_norm_relu(module: nn.Module) -> int:
    """Replace eligible BatchNorm2d + ReLU pairs after checkpoint loading."""
    replacements = 0
    for child in list(module.children()):
        replacements += fuse_batch_norm_relu(child)
    if not isinstance(module, nn.Sequential):
        return replacements

    names = list(module._modules)
    for batch_norm_name, relu_name in zip(names, names[1:]):
        batch_norm = module._modules[batch_norm_name]
        relu = module._modules[relu_name]
        if _is_standard_eval_batch_norm(batch_norm) and isinstance(relu, nn.ReLU):
            module._modules[batch_norm_name] = FusedBatchNormReLU(batch_norm)
            module._modules[relu_name] = nn.Identity()
            replacements += 1
    return replacements


def fuse_residual_batch_norm_add_relu(module: nn.Module) -> int:
    """Replace eligible eval RegNet blocks after checkpoint loading."""
    from nemotron_ocr.inference.models.detector.regnet import (
        ResBottleneckBlock,
    )

    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, ResBottleneckBlock) and not child.training:
            main_batch_norm = child.f.c[-1]
            projection_batch_norm = child.proj[-1] if child.proj is not None else None
            eligible = _is_standard_eval_batch_norm(main_batch_norm) and (
                projection_batch_norm is None
                or _is_standard_eval_batch_norm(projection_batch_norm)
            )
        else:
            eligible = False
        if eligible:
            module._modules[name] = FusedResidualBlock(child)
            replacements += 1
        else:
            replacements += fuse_residual_batch_norm_add_relu(child)
    return replacements


def fuse_aspp_batch_norm_relu_add(module: nn.Module) -> int:
    """Fuse safe eval-only residual ASPP paths after checkpoint loading."""
    from nemotron_ocr.inference.models.detector.aspp import ASPP

    replacements = 0
    for child in module.modules():
        if not isinstance(child, ASPP):
            continue
        batch_norm = child.final[1]
        relu = child.final[2]
        dropout = child.final[3]
        residual_channels_match = (
            child.final[0].out_channels == child.kernels[0][0].in_channels
        )
        if (
            child.training
            or not _is_standard_eval_batch_norm(batch_norm)
            or not isinstance(relu, nn.ReLU)
            or not isinstance(dropout, nn.Dropout)
            or dropout.p != 0
            or not residual_channels_match
        ):
            continue
        child._fused_norm_relu_add = FusedBatchNormReLUAdd(batch_norm)
        child.final[1] = nn.Identity()
        child.final[2] = nn.Identity()
        replacements += 1
    return replacements
