"""CPU-only tests for the faithful DRC reimplementation (src/groundup/drc.py).

These cover the contract of the architecture, not training behaviour: shapes
through D layers and N ticks, explicit state threading, gradient flow, that
the N internal ticks actually iterate, batch independence, and a parameter
count pinned to the verified faithful Sokoban DRC(3,3) retraining
(AlignmentResearch train-learned-planner run drc33/bkynosqi, reported as
1.29M parameters in Taufeeque et al. 2024).
"""

import pytest
import torch

from src.groundup.drc import DRCNet, DRCState, sokoban_drc33

torch.manual_seed(0)


def tiny_net(**overrides) -> DRCNet:
    """Small, fast DRC for CPU tests. Non-square board to catch H/W mixups."""
    kwargs = dict(
        observation_shape=(5, 6, 7),
        n_actions=9,
        depth=2,
        ticks=2,
        hidden_channels=8,
        encoder_channels=(8, 8),
        encoder_kernels=(4, 4),
        encoder_strides=(1, 1),
    )
    kwargs.update(overrides)
    torch.manual_seed(1234)
    return DRCNet(**kwargs)


def obs_batch(net: DRCNet, batch: int, seed: int = 7) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand((batch, *net.observation_shape), generator=gen)


class TestShapes:
    def test_output_and_state_shapes(self):
        net = tiny_net()
        obs = obs_batch(net, batch=3)
        logits, value, state = net(obs)
        assert logits.shape == (3, 9)
        assert value.shape == (3,)
        # (D, B, C, H, W); stride-1 SAME encoder preserves the 6x7 board.
        assert state.h.shape == (2, 3, 8, 6, 7)
        assert state.c.shape == (2, 3, 8, 6, 7)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()

    def test_strided_pixel_encoder_shapes(self):
        # The paper's 80x80 pixel encoder: (32,32) ch, (8,4) kernels, (4,2)
        # strides, SAME padding -> 10x10 recurrent state.
        net = tiny_net(
            observation_shape=(3, 80, 80),
            hidden_channels=8,
            encoder_channels=(8, 8),
            encoder_kernels=(8, 4),
            encoder_strides=(4, 2),
            encoder_relu=True,
        )
        obs = obs_batch(net, batch=2)
        logits, value, state = net(obs)
        assert state.h.shape == (2, 2, 8, 10, 10)
        assert logits.shape == (2, 9)
        assert value.shape == (2,)

    def test_paper_drc33_parameter_count(self):
        # 1,285,125 == exact parameter count of the faithful DRC(3,3)
        # (Taufeeque et al. 2024 report 1.29M). Breakdown in
        # docs/DrcFaithfulReimplementation.md.
        net = sokoban_drc33()
        n_params = sum(p.numel() for p in net.parameters())
        assert n_params == 1_285_125


class TestStateThreading:
    def test_fresh_state_is_deterministic(self):
        net = tiny_net()
        obs = obs_batch(net, batch=2)
        with torch.no_grad():
            logits_a, value_a, state_a = net(obs, state=None)
            logits_b, value_b, state_b = net(obs, state=net.initial_state(2))
        assert torch.equal(logits_a, logits_b)
        assert torch.equal(value_a, value_b)
        assert torch.equal(state_a.h, state_b.h)
        assert torch.equal(state_a.c, state_b.c)

    def test_carried_state_changes_output(self):
        net = tiny_net()
        obs = obs_batch(net, batch=2)
        with torch.no_grad():
            logits_1, _, state = net(obs)
            logits_2, _, state_2 = net(obs, state=state)
        assert not torch.allclose(logits_1, logits_2)
        assert not torch.allclose(state.h, state_2.h)

    def test_reset_state_zeroes_only_done_rows(self):
        net = tiny_net()
        obs = obs_batch(net, batch=2)
        with torch.no_grad():
            _, _, state = net(obs)
        done = torch.tensor([True, False])
        reset = DRCNet.reset_state(state, done)
        assert torch.equal(reset.h[:, 0], torch.zeros_like(reset.h[:, 0]))
        assert torch.equal(reset.c[:, 0], torch.zeros_like(reset.c[:, 0]))
        assert torch.equal(reset.h[:, 1], state.h[:, 1])
        assert torch.equal(reset.c[:, 1], state.c[:, 1])
        # A full reset reproduces the fresh-state output.
        all_done = torch.tensor([True, True])
        with torch.no_grad():
            logits_fresh, _, _ = net(obs)
            logits_reset, _, _ = net(obs, state=DRCNet.reset_state(state, all_done))
        assert torch.equal(logits_fresh, logits_reset)


class TestComputation:
    def test_gradients_flow_to_all_parameters(self):
        net = tiny_net()
        obs = obs_batch(net, batch=3)
        logits, value, state = net(obs)
        loss = logits.sum() + value.sum()
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert param.grad.abs().max() > 0, f"zero grad for {name}"

    def test_gradients_flow_through_carried_state(self):
        net = tiny_net()
        obs = obs_batch(net, batch=2)
        _, _, state = net(obs)
        logits2, value2, _ = net(obs, state=state)
        (logits2.sum() + value2.sum()).backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads and any(g.abs().max() > 0 for g in grads)

    def test_n_ticks_actually_iterate(self):
        net_1 = tiny_net(ticks=1)
        net_3 = tiny_net(ticks=3)
        # ticks does not change the parameter set; share the exact weights.
        net_3.load_state_dict(net_1.state_dict())
        obs = obs_batch(net_1, batch=2)
        with torch.no_grad():
            logits_1, _, state_1 = net_1(obs)
            logits_3, _, state_3 = net_3(obs)
        assert not torch.allclose(logits_1, logits_3)
        assert not torch.allclose(state_1.h, state_3.h)

    def test_depth_layers_are_distinct(self):
        net_1 = tiny_net(depth=1)
        net_2 = tiny_net(depth=2)
        assert sum(p.numel() for p in net_2.parameters()) > sum(p.numel() for p in net_1.parameters())

    def test_batch_independence(self):
        net = tiny_net()
        x0 = obs_batch(net, batch=1, seed=11)
        x1 = obs_batch(net, batch=1, seed=22)
        x2 = obs_batch(net, batch=1, seed=33)
        with torch.no_grad():
            logits_a, value_a, state_a = net(torch.cat([x0, x1]))
            logits_b, value_b, state_b = net(torch.cat([x0, x2]))
        # Row 0 must not depend on what else is in the batch.
        assert torch.allclose(logits_a[0], logits_b[0], atol=1e-6)
        assert torch.allclose(value_a[0], value_b[0], atol=1e-6)
        assert torch.allclose(state_a.h[:, 0], state_b.h[:, 0], atol=1e-6)
        # Sanity: the differing row does differ.
        assert not torch.allclose(logits_a[1], logits_b[1])


class TestVariantsAndAdaptations:
    def test_action_masking(self):
        net = tiny_net()
        obs = obs_batch(net, batch=2)
        mask = torch.ones(2, 9, dtype=torch.bool)
        mask[0, :5] = False
        mask[1, 8] = False
        with torch.no_grad():
            logits, _, _ = net(obs, action_mask=mask)
        probs = torch.softmax(logits, dim=-1)
        assert probs[0, :5].max() < 1e-8
        assert probs[1, 8] < 1e-8
        assert torch.isfinite(probs).all()

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(output_activation="sigmoid"),
            dict(fence="same"),
            dict(fence="none"),
            dict(pool_projection="full"),
            dict(pool_and_inject=False),
            dict(encoder_channels=(), encoder_kernels=(), encoder_strides=(), encoder_skip=False),
            dict(encoder_skip=False),
        ],
    )
    def test_documented_variants_run(self, overrides):
        net = tiny_net(**overrides)
        obs = obs_batch(net, batch=2)
        logits, value, state = net(obs)
        assert logits.shape == (2, 9)
        assert value.shape == (2,)
        assert torch.isfinite(logits).all()
        (logits.sum() + value.sum()).backward()

    def test_variant_flags_change_the_function(self):
        base = tiny_net()
        sig = tiny_net(output_activation="sigmoid")
        sig.load_state_dict(base.state_dict())  # same weights, different gate
        obs = obs_batch(base, batch=2)
        with torch.no_grad():
            logits_base, _, _ = base(obs)
            logits_sig, _, _ = sig(obs)
        assert not torch.allclose(logits_base, logits_sig)

    def test_encoder_skip_channel_mismatch_raises(self):
        with pytest.raises(ValueError, match="encoder_skip"):
            tiny_net(encoder_channels=(8, 4), hidden_channels=8)

    def test_drc11_reduces_to_single_module_single_tick(self):
        net = tiny_net(depth=1, ticks=1)
        obs = obs_batch(net, batch=2)
        logits, value, state = net(obs)
        assert state.h.shape[0] == 1
        assert logits.shape == (2, 9)
