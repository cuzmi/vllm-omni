# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.flux2.flux2_transformer import (
    Flux2PosEmbed,
    Flux2RopePrepare,
    Flux2Transformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _rope_model():
    # No weights, process groups or attention backend are needed to test the
    # position preparation contract and forward dispatch.
    model = Flux2Transformer2DModel.__new__(Flux2Transformer2DModel)
    nn.Module.__init__(model)
    model.parallel_config = DiffusionParallelConfig()
    model.rope_prepare = Flux2RopePrepare(Flux2PosEmbed(theta=2000, axes_dim=(2, 2, 2, 2)))
    return model


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("id_dtype", [torch.float32, torch.float64, torch.int64])
def test_prepared_rope_matches_original_operations(batched, id_dtype):
    model = _rope_model()
    img_ids = torch.arange(28).reshape(7, 4).to(id_dtype)
    txt_ids = torch.arange(12).reshape(3, 4).to(id_dtype)
    txt_cos, txt_sin, img_cos, img_sin = model.rope_prepare(img_ids, txt_ids)
    expected = (torch.cat([txt_cos, img_cos]), torch.cat([txt_sin, img_sin]))
    actual = model.prepare_rotary_emb(
        img_ids.unsqueeze(0) if batched else img_ids,
        txt_ids.unsqueeze(0) if batched else txt_ids,
    )
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_same_shape_different_positions_do_not_reuse_stale_values():
    model = _rope_model()
    img_ids = torch.zeros(1, 7, 4)
    txt_ids = torch.zeros(1, 3, 4)
    first = model.prepare_rotary_emb(img_ids, txt_ids)
    second = model.prepare_rotary_emb(img_ids + 1, txt_ids)
    negative = model.prepare_rotary_emb(img_ids, txt_ids + 2)
    assert not torch.equal(first[0][3:], second[0][3:])
    assert not torch.equal(first[0][:3], negative[0][:3])
    torch.testing.assert_close(first[0][:3], second[0][:3], rtol=0, atol=0)
    torch.testing.assert_close(first[0][3:], negative[0][3:], rtol=0, atol=0)


class _RopeConsumer(nn.Module):
    """Small deterministic consumer; this does not test production attention."""

    def forward(self, hidden_states, encoder_hidden_states, image_rotary_emb, **kwargs):
        cos, sin = image_rotary_emb
        text_len = encoder_hidden_states.shape[1]
        # Make the output depend on both image and text positions so mixing
        # positive/negative embeddings cannot silently pass the parity check.
        offset = cos[text_len:] + sin[text_len:] + cos[:text_len].mean(0)
        return encoder_hidden_states, hidden_states + offset.to(hidden_states.dtype)


def _forward_model():
    model = _rope_model()
    model.time_guidance_embed = Mock(return_value=torch.zeros(1, 4))
    model.double_stream_modulation_img = Mock(return_value=None)
    model.double_stream_modulation_txt = Mock(return_value=None)
    model.single_stream_modulation = Mock(return_value=(None,))
    model.x_embedder = nn.Identity()
    model.context_embedder = nn.Identity()
    model.transformer_blocks = nn.ModuleList([_RopeConsumer()])
    model.single_transformer_blocks = nn.ModuleList()
    model.norm_out = Mock(side_effect=lambda hidden, temb: hidden)
    model.proj_out = nn.Identity()
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_reuses_prepared_rope_without_changing_outputs(dtype):
    model = _forward_model()
    kwargs = dict(
        hidden_states=torch.arange(28).reshape(1, 7, 4).to(dtype),
        encoder_hidden_states=torch.zeros(1, 3, 4, dtype=dtype),
        timestep=torch.ones(1),
        img_ids=torch.arange(28).reshape(1, 7, 4),
        txt_ids=torch.arange(12).reshape(1, 3, 4),
        return_dict=False,
    )
    prepared = model.prepare_rotary_emb(kwargs["img_ids"], kwargs["txt_ids"])
    calls = []
    handle = model.rope_prepare.register_forward_hook(lambda *args: calls.append(1))
    for step in range(4):
        kwargs["timestep"] = torch.tensor([1.0 - step / 4])
        expected = model(**kwargs)[0]
        actual = model(**kwargs, image_rotary_emb=prepared)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    handle.remove()
    assert len(calls) == 4  # Only uncached reference calls compute RoPE.


def test_sequence_parallel_does_not_bypass_rope_hooks():
    model = _forward_model()
    model.parallel_config = DiffusionParallelConfig(ulysses_degree=2)
    calls = []
    handle = model.rope_prepare.register_forward_hook(lambda *args: calls.append(1))
    model(
        hidden_states=torch.zeros(1, 7, 4),
        encoder_hidden_states=torch.zeros(1, 3, 4),
        timestep=torch.ones(1),
        img_ids=torch.zeros(1, 7, 4),
        txt_ids=torch.zeros(1, 3, 4),
        # Invalid precomputed shapes would break the consumer if used.
        image_rotary_emb=(torch.empty(0), torch.empty(0)),
        return_dict=False,
    )
    handle.remove()
    assert len(calls) == 1
