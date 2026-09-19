import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agent import Agent  # noqa: E402


class AdaptiveAttackTest(unittest.TestCase):
    def setUp(self):
        self.agent = Agent.__new__(Agent)
        self.agent.args = SimpleNamespace(
            avg_align_topk_ratio=0.5,
            sparsity=0.5,
            adaptive_sign_temperature=0.1,
            adaptive_cosine_floor=0.05,
            adaptive_mdf_margin=0.0,
            median_guard_two_sided=False,
        )

    def test_differentiable_update_matches_uploaded_state_shape(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 3, bias=False),
            torch.nn.BatchNorm1d(3),
        )
        initial = Agent.get_model_state(model).detach().clone()
        with torch.no_grad():
            model[0].weight.add_(0.25)

        update = Agent._differentiable_state_update(model, initial)

        self.assertEqual(update.numel(), initial.numel())
        self.assertTrue(update.requires_grad)
        update.square().sum().backward()
        self.assertIsNotNone(model[0].weight.grad)

    def test_pdc_loss_prefers_clean_pair_statistics(self):
        clean_self = torch.tensor([5.0, 4.0, -3.0, 2.0, 0.2, -0.1])
        clean_peer = torch.tensor([4.0, -5.0, -2.0, 3.0, -0.1, 0.2])

        matched = clean_self.clone().requires_grad_(True)
        matched_loss = self.agent._pdc_adaptive_loss(
            matched, clean_self, [clean_peer], [clean_peer]
        )

        collapsed = clean_peer.clone().requires_grad_(True)
        collapsed_loss = self.agent._pdc_adaptive_loss(
            collapsed, clean_self, [clean_peer], [clean_peer]
        )

        self.assertLess(matched_loss.item(), collapsed_loss.item())
        collapsed_loss.backward()
        self.assertTrue(torch.isfinite(collapsed.grad).all())

    def test_pdc_single_attacker_still_uses_clean_anchor(self):
        reference = torch.tensor([3.0, -2.0, 1.0, -0.5])
        close = reference.clone().requires_grad_(True)
        opposite = (-reference).clone().requires_grad_(True)

        close_loss = self.agent._pdc_adaptive_loss(close, reference, [], [])
        opposite_loss = self.agent._pdc_adaptive_loss(opposite, reference, [], [])

        self.assertLess(close_loss.item(), opposite_loss.item())

    def test_mdf_wrong_sign_has_finite_nonzero_gradient(self):
        clean = torch.linspace(0.1, 1.0, 100)
        median_proxy = clean.clone()
        current = (-0.01 * torch.ones(100)).requires_grad_(True)

        cosine_loss, sign_loss = self.agent._mdf_losses(
            current, clean, median_proxy
        )
        (cosine_loss + sign_loss).backward()

        self.assertGreater(sign_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertGreater(current.grad.norm().item(), 0.0)

    def test_pdc_zero_update_gradient_is_bounded(self):
        current = torch.zeros(100, requires_grad=True)
        reference = torch.linspace(-1.0, 1.0, 100)

        loss = self.agent._pdc_adaptive_loss(current, reference, [], [])
        loss.backward()

        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertLess(current.grad.abs().max().item(), 100.0)

    def test_mdf_matches_clean_score_instead_of_forcing_one_direction(self):
        clean = torch.tensor([5.0, 4.0, -3.0, 2.0, 0.2, -0.1])
        median_proxy = torch.tensor([4.0, 3.0, -2.0, -1.0, 0.1, 0.2])
        matched = clean.clone().requires_grad_(True)
        opposite = (-clean).clone().requires_grad_(True)

        matched_losses = self.agent._mdf_losses(matched, clean, median_proxy)
        opposite_losses = self.agent._mdf_losses(opposite, clean, median_proxy)

        self.assertLess(
            sum(loss.item() for loss in matched_losses),
            sum(loss.item() for loss in opposite_losses),
        )

    def test_local_train_returns_defense_shaped_adaptive_update(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
        )
        dataset = TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,)))
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        agent = Agent.__new__(Agent)
        agent.id = 0
        agent.is_malicious = True
        agent.train_loader = loader
        agent.adaptive_clean_train_loader = loader
        agent._adaptive_reference_cache = {}
        agent.args = SimpleNamespace(
            attack="adaptive_pdc",
            num_corrupt=1,
            cease_poison=100,
            client_lr=0.01,
            lr_decay=1.0,
            wd=0.0,
            momentum=0.0,
            local_ep=1,
            device=torch.device("cpu"),
            lambda_cos=1.0,
            lambda_sign=1.0,
            lambda_div=1.0,
            avg_align_topk_ratio=0.5,
            sparsity=0.5,
            adaptive_sign_temperature=0.1,
            adaptive_cosine_floor=0.05,
            adaptive_mdf_margin=0.0,
            median_guard_two_sided=False,
        )

        update = agent.local_train(
            model,
            torch.nn.CrossEntropyLoss(),
            round=1,
            adaptive_mode="refine",
            adaptive_peer_updates=[],
            adaptive_peer_references=[],
            return_state_update=True,
        )

        self.assertEqual(update.numel(), Agent.get_model_state(model).numel())
        self.assertTrue(torch.isfinite(update).all())


if __name__ == "__main__":
    unittest.main()
