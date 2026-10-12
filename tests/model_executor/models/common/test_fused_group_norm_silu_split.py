# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Large-group regressions for the multi-CTA GN+SiLU path."""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.ops import fused_group_norm_silu

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def reference(x, weight, bias):
    return F.silu(F.group_norm(x.float(), 32, weight.float(), bias.float(), 1e-6))


@pytest.mark.parametrize("shape", [(1, 128, 1, 256, 256), (1, 128, 257, 257), (2, 64, 256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_large_groups_and_partial_tail(shape, dtype):
    torch.manual_seed(20261011)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[1], device="cuda", dtype=dtype)
    bias = torch.randn_like(weight)
    out = fused_group_norm_silu(x, weight, bias)
    ref = reference(x, weight, bias).to(dtype)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (2e-3, 2e-3) if dtype == torch.float16 else (1e-2, 1e-2)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("permuted", [False, True])
@torch.inference_mode()
def test_large_groups_autocast(permuted):
    x = torch.randn(1, 128, 256, 256, device="cuda", dtype=torch.float16)
    if permuted:
        x = x.transpose(-1, -2)
    weight = torch.randn(128, device="cuda")
    bias = torch.randn_like(weight)
    with torch.autocast("cuda", dtype=torch.float16):
        out = fused_group_norm_silu(x, weight, bias)
        ref = F.silu(F.group_norm(x, 32, weight, bias, 1e-6))
    assert out.dtype == ref.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_large_groups_offset_stability():
    x = torch.randn(1, 128, 256, 256, device="cuda") * 0.1 + 10000
    weight = torch.randn(128, device="cuda")
    bias = torch.randn_like(weight)
    out = fused_group_norm_silu(x, weight, bias)
    ref = reference(x, weight, bias)
    assert out.isfinite().all()
    # Same tolerance as the existing large-offset GN regression; FP32 mean
    # rounding is amplified by the inverse of the small standard deviation.
    torch.testing.assert_close(out, ref, rtol=0.05, atol=0.1)


@torch.inference_mode()
def test_large_groups_graph_replay_recomputes_statistics():
    x = torch.randn(1, 128, 256, 256, device="cuda")
    weight = torch.randn(128, device="cuda")
    bias = torch.randn_like(weight)
    for _ in range(3):
        fused_group_norm_silu(x, weight, bias)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_group_norm_silu(x, weight, bias)
    for scale in [0.1, 5.0]:
        x.normal_().mul_(scale).add_(3.0)
        graph.replay()
        torch.testing.assert_close(out, reference(x, weight, bias), rtol=1e-5, atol=1e-5)
