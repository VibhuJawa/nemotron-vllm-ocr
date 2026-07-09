# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import nemotron_ocr.inference.models.relational as relational
from nemotron_ocr.inference.models.relational import (
    GlobalRelationalModel,
    _pad_flat_to_batched,
    get_directions,
)


def _make_model(**kwargs):
    return GlobalRelationalModel(
        num_input_channels=[4],
        recog_feature_depth=4,
        k=4,
        num_layers=1,
        **kwargs,
    )


def _distance_matrix(quads):
    distances = torch.cdist(quads.mean(dim=1), quads.mean(dim=1))
    distances.fill_diagonal_(torch.inf)
    return distances


def _legacy_geometry_inputs(model, proj_rects, mid_pts, quads, counts):
    feat_dim = proj_rects.shape[1]
    z = model.k - 1
    seq_len = z + 1
    n_total = proj_rects.shape[0]
    enc_input = torch.zeros(n_total, seq_len, 2 * feat_dim + 2)
    mask = torch.ones(n_total, seq_len, dtype=torch.bool)
    closest = torch.zeros(n_total, seq_len, dtype=torch.long)

    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)

    for image_index, count in enumerate(counts):
        if count == 0:
            continue
        start, end = offsets[image_index : image_index + 2]
        rects = proj_rects[start:end]
        centers = mid_pts[start:end]
        image_quads = quads[start:end]
        z_i = min(count - 1, z)

        from_rects = rects.unsqueeze(1).expand(-1, seq_len, -1)
        enc_input[start:end, 0, :feat_dim] = rects
        enc_input[start:end, 0, 2 * feat_dim] = -1
        enc_input[start:end, 0, 2 * feat_dim + 1] = -2
        mask[start:end, 0] = False

        if z_i == 0:
            continue
        topk_d, topk_idx = torch.topk(
            _distance_matrix(image_quads),
            k=z_i,
            dim=1,
            largest=False,
            sorted=False,
        )
        neighbor_rects = torch.gather(
            rects.unsqueeze(0).expand(count, -1, -1),
            dim=1,
            index=topk_idx.unsqueeze(2).expand(-1, -1, feat_dim),
        )
        neighbor_centers = torch.gather(
            centers.unsqueeze(0).expand(count, -1, -1),
            dim=1,
            index=topk_idx.unsqueeze(2).expand(-1, -1, 2),
        )
        directions = get_directions(image_quads, neighbor_centers)

        enc_input[start:end, 1 : z_i + 1, :feat_dim] = from_rects[
            :, 1 : z_i + 1
        ]
        enc_input[
            start:end, 1 : z_i + 1, feat_dim : 2 * feat_dim
        ] = neighbor_rects
        enc_input[start:end, 1 : z_i + 1, 2 * feat_dim] = topk_d
        enc_input[start:end, 1 : z_i + 1, 2 * feat_dim + 1] = directions
        mask[start:end, 1 : z_i + 1] = False
        closest[start:end, 1 : z_i + 1] = topk_idx + 1

    return enc_input, mask, closest


def test_pad_flat_to_batched_preserves_ragged_order_without_scalar_reads():
    flat = torch.arange(9 * 2, dtype=torch.float32).reshape(9, 2)
    counts = torch.tensor([3, 0, 2, 4], dtype=torch.long)

    actual = _pad_flat_to_batched(flat, counts, k_max=4, pad_value=-1)

    expected = torch.full((4, 4, 2), -1, dtype=torch.float32)
    expected[0, :3] = flat[:3]
    expected[2, :2] = flat[3:5]
    expected[3, :4] = flat[5:]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_batched_geometry_exactly_matches_per_image_topk_with_ties(monkeypatch):
    counts = torch.tensor([4, 2, 4, 1, 0], dtype=torch.long)
    counts_list = counts.tolist()
    n_total = sum(counts_list)
    proj_rects = torch.arange(n_total * 5, dtype=torch.float32).reshape(
        n_total, 5
    ) / 17
    mid_pts = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
            [9.0, 0.0],
            [10.0, 2.0],
            [20.0, 0.0],
        ]
    )
    quads = mid_pts[:, None, :].expand(-1, 4, -1).clone()
    quads[:, (0, 3), 0] -= 0.1
    quads[:, (1, 2), 0] += 0.1

    calls = []

    def fake_batched_cdist(padded_quads, region_counts, *args, **kwargs):
        calls.append((padded_quads.shape, region_counts.clone()))
        batch_size, k_max = padded_quads.shape[:2]
        output = padded_quads.new_zeros(batch_size, k_max, k_max)
        for image_index, count in enumerate(region_counts.tolist()):
            output[image_index, :count, :count] = _distance_matrix(
                padded_quads[image_index, :count]
            )
        return output

    monkeypatch.setattr(relational, "get_cdist_batched", fake_batched_cdist)
    model = _make_model(batched_geometry=True)

    actual = model._build_batched_geometry_inputs(
        proj_rects,
        mid_pts,
        quads,
        counts,
        counts_list,
    )
    expected = _legacy_geometry_inputs(
        model, proj_rects, mid_pts, quads, counts_list
    )

    assert len(calls) == 1
    assert calls[0][0] == torch.Size([5, 4, 4, 2])
    torch.testing.assert_close(calls[0][1], counts, rtol=0, atol=0)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=0, atol=0
        )


def test_batched_geometry_is_default_off_and_environment_opt_in(monkeypatch):
    monkeypatch.delenv("NEMOTRON_OCR_BATCHED_RELATIONAL_GEOMETRY", raising=False)
    assert not _make_model().batched_geometry

    monkeypatch.setenv("NEMOTRON_OCR_BATCHED_RELATIONAL_GEOMETRY", "true")
    assert _make_model().batched_geometry
