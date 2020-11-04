"""
Unit Tests for PyTorch optimizer Package
(c) 2020 d373c7
"""
import unittest
from typing import Any
import torch
import torch.nn as nn
import torch.optim as opt
from d373c7.pytorch.optimizer import AdamWOptimizer


class TestNN(nn.Module):
    def _forward_unimplemented(self, *inp: Any) -> None:
        pass

    def __init__(self):
        super(TestNN, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


class TestAdamW(unittest.TestCase):
    def test_create_base(self):
        m = TestNN()
        o = AdamWOptimizer(m, 0.1)
        self.assertIsInstance(o, AdamWOptimizer, f'Not expected class {type(o)}')
        self.assertIsInstance(o.optimizer, opt.AdamW, f'Was expecting to get a AdamW optimizer {o.optimizer}')

    def test_zero_grad(self):
        m = TestNN()
        o = AdamWOptimizer(m, 0.1)
        t = torch.rand(20, 10)
        r = torch.mean(m(t))
        self.assertIsNone(m.linear.weight.grad, f'Gradients should not be set')
        r.backward()
        self.assertEqual(m.linear.weight.grad.shape, (1, 10), f'Gradients should have been set')
        o.zero_grad()
        self.assertEqual(torch.sum(m.linear.weight.grad), 0, f'Gradients should not be all 0')

    def test_step(self):
        m = TestNN()
        o = AdamWOptimizer(m, 0.1)
        t = torch.rand(20, 10)
        r = torch.mean(m(t))
        w = m.linear.weight.detach().numpy().tolist()
        self.assertIsNone(m.linear.weight.grad, f'Gradients should not be set')
        r.backward()
        o.step()
        self.assertNotEqual(m.linear.weight.detach().numpy().tolist(), w, f'weights should have changed')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
