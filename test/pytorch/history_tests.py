"""
Unit Tests for PyTorch History Package
(c) 2020 d373c7
"""
import unittest
import torch
import torch.utils.data as data
# noinspection PyProtectedMember
from d373c7.pytorch.common import _History
# inspection PyProtectedMember
from math import ceil

FILES_DIR = './files/'


class BaseHistoryCases(unittest.TestCase):
    def test_create_base(self):
        bs = 1
        t = torch.Tensor([[0.0, 0.1], [0.2, 0.3]])
        ds = data.TensorDataset(t)
        dl = data.DataLoader(ds, batch_size=bs)
        h = _History(dl)
        self.assertEqual(h.samples, t.shape[0], f'Sample size should have been {t.shape[0]}')
        self.assertEqual(h.batch_size, bs, f'Batch size should have been {bs}')
        self.assertEqual(h.epoch, 0, f'Initial epoch should be 0. Was {h.epoch}')
        self.assertEqual(h.steps, ceil(h.samples/bs), f'Number of steps should have been. {ceil(h.samples/bs)}')
        self.assertEqual(h.step, 0, f'Current step should be 0. Got {h.step}')
        # self.assertIsInstance(h.history, Dict, f'History object should have been a Dict {type(h.history)}')
        # self.assertListEqual(sorted(list(h.history.keys())), sorted(metrics), 'History object have metric keys')
        # for k, v in h.history.items():
        #    self.assertIsInstance(v, List, f'Metric values should be a list {type(v)}')
        #    self.assertEqual(len(v), 0, f'Metric values should have been empty')

    def test_work(self):
        bs = 1
        t = torch.Tensor([[0.0, 1.0], [0.0, 1.0]])
        ds = data.TensorDataset(t)
        dl = data.DataLoader(ds, batch_size=bs)
        h = _History(dl)
        h.start_epoch()
        self.assertEqual(h.epoch, 1, f'Current Epoch should have been 1. Was {h.epoch}')
        self.assertEqual(h.step, 0, f'Current step should have been 0. Was {h.step}')
        h.start_step()
        self.assertEqual(h.epoch, 1, f'Current Epoch should have been 1. Was {h.epoch}')
        self.assertEqual(h.step, 1, f'Current Step should have been 1. Was {h.step}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
