"""
Unit Tests for PyTorch schedule Package
(c) 2020 d373c7
"""
import unittest
import torch
import torch.nn as nn
import torch.utils.data as data
from d373c7.pytorch.optimizer import AdamWOptimizer
from d373c7.pytorch.schedule import LRHistory, LinearLR
from typing import Any, List, Dict


class TestNN(nn.Module):
    def _forward_unimplemented(self, *inp: Any) -> None:
        pass

    def __init__(self):
        super(TestNN, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


class TestLinearLR(unittest.TestCase):
    def test_create_base(self):
        n = 3
        base_lr = 10e-2
        end_lr = 10e-3
        m = TestNN()
        o = AdamWOptimizer(m, base_lr)
        s = LinearLR(end_lr, n, o)
        self.assertIsInstance(s, LinearLR, f'Expected a LinearLR class')
        self.assertIsInstance(s.get_lr(), List, f'get_lr should have returned a List')
        self.assertEqual(s.get_lr()[0], base_lr, f'First LR should have been the base LR. Got {s.get_lr()[0]}')
        self.assertEqual(s.get_lr()[0], o.lr, f'First LR should have been the same as opt LR Got {s.get_lr()[0]}')

    def test_work_step(self):
        n = 3
        base_lr = 10e-2
        end_lr = 10e-3
        m = TestNN()
        o = AdamWOptimizer(m, base_lr)
        s = LinearLR(end_lr, n, o)
        o.step()
        s.step()
        self.assertEqual(s.get_lr()[0], base_lr - ((base_lr - end_lr) / (n-1)),
                         f'Incorrect LR. Hoping for {base_lr - ((base_lr - end_lr) / (n-1))}')


class TestLRHistory(unittest.TestCase):
    def test_create_base(self):
        bs = 1
        m = TestNN()
        o = AdamWOptimizer(m, 10e-2)
        s = LinearLR(10e-1, 100, o)
        t = torch.Tensor([[0.0, 0.1], [0.2, 0.3]])
        ds = data.TensorDataset(t)
        dl = data.DataLoader(ds, batch_size=bs)
        history = LRHistory(dl, s, 5, 0.1, 20)
        self.assertIsInstance(history, LRHistory, f'Did not get correct type. Got {type(history)}')
        self.assertEqual(history.batch_size, bs, f'Incorrect batch size. {history.batch_size}')
        self.assertEqual(history.samples, t.shape[0], f'Incorrect number of samples {history.samples}')
        self.assertEqual(history.steps, t.shape[0]/bs, f'Incorrect number of steps. Expecting {t.shape[0]/bs}')
        self.assertIsInstance(history.history, Dict, f'history should have been a Dict. got {type(history.history)}')
        self.assertEqual(history.epoch, 0, f'Epoch should have been 0. Got {history.epoch}')
        self.assertEqual(history.step, 0, f'Should have been at step 1. Got {history.step}')
        self.assertEqual(history.early_break(), False, f'Early break not False')

    def test_work(self):
        bs = 1
        m = TestNN()
        o = AdamWOptimizer(m, 10e-2)
        s = LinearLR(10e-1, 100, o)
        t = torch.Tensor([[0.0, 0.1], [0.2, 0.3]])
        ds = data.TensorDataset(t)
        dl = data.DataLoader(ds, batch_size=bs)
        history = LRHistory(dl, s, 5, 0.1, 20)
        history.start_epoch()
        self.assertEqual(history.epoch, 1, f'Epoch should have been 1. Got {history.epoch}')
        history.start_step()
        self.assertEqual(history.step, 1, f'Step should have been 1. Got {history.step} ')
        loss = 0.5
        history.end_step(torch.Tensor([1]), [torch.Tensor([1])], torch.Tensor([loss]))
        h = history.history
        self.assertEqual(len(h[history.loss_key]), 1, f'Should be a list of 1. Got {len(h[history.loss_key])}')
        self.assertEqual(len(h[history.lr_key]), 1, f'Should be a list of 1. Got {len(h[history.loss_key])}')
        self.assertEqual(h[history.loss_key][0], loss, f'First element should not the loss {h[history.loss_key][0]}')
        self.assertEqual(h[history.lr_key][0], o.lr, f'First element should have been the lr {h[history.lr_key][0]}')
        self.assertEqual(history.early_break(), False, f'Early Break should have been False')
        history.start_step()
        self.assertEqual(history.epoch, 1, f'Epoch should still be 1. Got {history.epoch}')
        self.assertEqual(history.step, 2, f'Step should have been 2. Got {history.step}')
        self.assertEqual(history.early_break(), False, f'Early Break should have been False')
        with self.assertRaises(NotImplementedError):
            history.end_epoch()

    def test_early_break_diverge(self):
        bs = 1
        m = TestNN()
        o = AdamWOptimizer(m, 10e-2)
        s = LinearLR(10e-1, 100, o)
        t = torch.Tensor([[0.0, 0.1], [0.2, 0.3]])
        ds = data.TensorDataset(t)
        dl = data.DataLoader(ds, batch_size=bs)
        loss = 0.5
        diverge = 5
        # Smooth set to 1 for ease of testing
        history = LRHistory(dl, s, diverge, 1, 20)
        history.start_epoch()
        history.start_step()
        history.end_step(torch.Tensor([1]), [torch.Tensor([1])], torch.Tensor([loss]))
        self.assertEqual(history.early_break(), False, f'Early Break should have been false')
        history.start_step()
        history.end_step(torch.Tensor([1]), [torch.Tensor([1])], torch.Tensor([loss * (diverge - 0.01)]))
        self.assertEqual(history.early_break(), False, f'Early Break should have been false')
        history.start_step()
        history.end_step(torch.Tensor([1]), [torch.Tensor([1])], torch.Tensor([(loss * diverge) + 0.01]))
        self.assertEqual(history.early_break(), True, f'Early Break should have been false')

    def test_early_break_max_steps(self):
        bs = 1
        m = TestNN()
        o = AdamWOptimizer(m, 10e-2)
        s = LinearLR(10e-1, 100, o)
        t = torch.Tensor([[0.0, 0.1], [0.2, 0.3]])
        ds = data.TensorDataset(t)
        dl = data.DataLoader(ds, batch_size=bs)
        loss = 0.5
        diverge = 5
        history = LRHistory(dl, s, diverge, 0.1, 2)
        history.start_epoch()
        history.start_step()
        history.end_step(torch.Tensor([1]), [torch.Tensor([1])], torch.Tensor([loss]))
        self.assertEqual(history.early_break(), False, f'Early Break should have been false')
        history.start_step()
        history.end_step(torch.Tensor([1]), [torch.Tensor([1])], torch.Tensor([loss]))
        self.assertEqual(history.early_break(), True, f'Early Break should have been True')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
