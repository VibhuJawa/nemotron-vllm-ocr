# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import nemotron_ocr.inference.pipeline_v2 as pipeline_v2
from nemotron_ocr.inference.pipeline import NMS_PROB_THRESHOLD
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2


def _pipeline(peak_kernel=3):
    pipeline = NemotronOCRV2.__new__(NemotronOCRV2)
    object.__setattr__(pipeline, "_prefilter_peak_kernel", peak_kernel)
    return pipeline


def _legacy_prefilter(det_conf, det_rboxes, peak_kernel):
    d_top = det_rboxes[..., 0].float()
    d_right = det_rboxes[..., 1].float()
    d_bottom = det_rboxes[..., 2].float()
    d_left = det_rboxes[..., 3].float()

    lr_min = torch.minimum(d_left, d_right)
    lr_max = torch.maximum(d_left, d_right).clamp(min=1.0)
    tb_min = torch.minimum(d_top, d_bottom)
    tb_max = torch.maximum(d_top, d_bottom).clamp(min=1.0)
    centerness = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))

    adjusted = torch.sigmoid(det_conf.float()) * centerness
    pooled = F.max_pool2d(
        adjusted.unsqueeze(1),
        peak_kernel,
        stride=1,
        padding=peak_kernel // 2,
    )
    pooled = pooled[:, 0, : det_conf.shape[1], : det_conf.shape[2]]
    peaks = (adjusted == pooled) & (adjusted > NMS_PROB_THRESHOLD)

    filtered = det_conf.clone()
    filtered[~peaks] = -100.0
    return filtered


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16]
)
def test_prefilter_reused_probability_matches_legacy_dense_sigmoid(dtype):
    generator = torch.Generator().manual_seed(7)
    det_conf = torch.randn((2, 9, 11), generator=generator).to(dtype)
    det_rboxes = (
        torch.rand((2, 9, 11, 5), generator=generator).mul_(80).add_(0.25)
    ).to(dtype)

    expected_probability = torch.sigmoid(
        _legacy_prefilter(det_conf, det_rboxes, peak_kernel=3)
    )

    original_logits = det_conf.clone()
    actual_probability = _pipeline()._prefilter_detections(det_conf, det_rboxes)

    assert actual_probability.dtype == expected_probability.dtype
    assert torch.equal(actual_probability, expected_probability)
    assert torch.equal(det_conf, original_logits)


def test_run_nms_forwards_precomputed_probability_without_recomputing(monkeypatch):
    supplied_probability = torch.rand((1, 2, 3), dtype=torch.float32)
    captured = {}

    def fake_rrect_to_quads(det_rboxes, downsample):
        return torch.zeros((*det_rboxes.shape[:-1], 4, 2), dtype=torch.float32)

    def fake_nms(coords, probability, **kwargs):
        captured["probability"] = probability
        return (
            torch.zeros((1, 4, 2), dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.int64),
        )

    monkeypatch.setattr(pipeline_v2, "rrect_to_quads", fake_rrect_to_quads)
    monkeypatch.setattr(pipeline_v2, "quad_non_maximal_suppression", fake_nms)

    result = _pipeline()._run_nms(
        torch.randn((1, 2, 3)),
        torch.ones((1, 2, 3, 5)),
        supplied_probability,
    )

    assert captured["probability"] is supplied_probability
    assert result[3] is supplied_probability
