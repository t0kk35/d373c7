"""
Unit Tests for PyTorch Classifier Model tests
(c) 2020 d373c7
"""
import unittest
import torch
import torch.utils.data as data
import torch.nn.functional as tf
import d373c7.pytorch.models
from d373c7.pytorch.models.classifiers import BinaryClassifierHistory


class BinaryClassifierHistoryTests(unittest.TestCase):
    def test_create(self):
        bs = 2
        acc_key = 'acc'
        loss_key = 'loss'
        # Tensor with 4 samples
        t0 = torch.Tensor([[0.0], [1.0], [2.0], [3.0]])
        # 2 Tensors to use as labels for a step. Must be 2 as samples / bs = 2.
        t1 = torch.Tensor([[0.0], [1.0]])
        t2 = torch.Tensor([[0.1], [1.0]])
        ds = data.TensorDataset(t0)
        dl = data.DataLoader(ds, batch_size=bs)
        ls = tf.binary_cross_entropy(t1, t1)
        h = BinaryClassifierHistory(dl)
        h.start_epoch()
        h.start_step()
        h.end_step(t1, [t1], ls)
        s1 = h.step_stats()
        self.assertEqual(h.early_break(), False, f'Early Break should be False')
        self.assertEqual(h.step, 1, f'Step count should have been one {h.step}')
        self.assertEqual(s1[acc_key], 1.0, f'Expected the accuracy of this step to be 1. Got {s1[acc_key]}')
        self.assertEqual(s1[loss_key], 0.0, f'Expected loss of this step to be 0.0. Got {s1[loss_key]}')
        ls = tf.binary_cross_entropy(t1, t2)
        h.start_step()
        h.end_step(t1, [t2], ls)
        s2 = h.step_stats()
        ea = 3 / h.samples
        el = 0 + ls.item() / h.steps
        self.assertEqual(h.early_break(), False, f'Early Break should be False')
        self.assertEqual(h.step, 2, f'Step count should have been two {h.step}')
        self.assertEqual(s2[acc_key], ea, f'Expected the accuracy of this step to be {ea}. Got {s2[acc_key]}')
        self.assertEqual(s2[loss_key], el, f'Expected loss of this step to be {el}. Got {s2[loss_key]}')
        h.end_epoch()
        hd = h.history
        self.assertEqual(h.early_break(), False, f'Early Break should be False')
        self.assertEqual(hd[acc_key][0], ea, f'Expected the accuracy of the epoch to be {ea}. Got {hd[acc_key][0]}')
        self.assertEqual(hd[loss_key][0], el, f'Expected loss of the epoch to be {el}. Got {hd[loss_key][0]}')


class BinaryFraudClassifierTests(unittest.TestCase):
    def test_create(self):
        pass


def main():
    unittest.main()


if __name__ == '__main__':
    main()
