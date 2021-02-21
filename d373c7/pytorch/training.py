"""
Imports for Pytorch Training module
(c) 2020 d373c7
"""
import logging
import os
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
from .common import _History
from .loss import _LossBase
from .optimizer import _Optimizer
from .schedule import LRHistory, LinearLR
# noinspection PyProtectedMember
from .models.common import _Model, _ModelManager
# inspection PyProtectedMember
from typing import Tuple, Dict, List


logger = logging.getLogger(__name__)


class Trainer(_ModelManager):
    """Class to train a Neural net. Embeds some methods that hide the Pytorch training logic/loop.

    :argument model: The model to be trained. This needs to be a d373c7 model. Not a regular nn.Module.
    :argument device: A torch device (CPU or GPU) to use during training.
    :argument train_dl: A torch DataLoader object containing the training data.
    :argument val_dl: A torch DataLoader object containing the validation data.
    """
    def __init__(self, model: _Model, device: torch.device, train_dl: data.DataLoader, val_dl: data.DataLoader):
        _ModelManager.__init__(self, model, device)
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._train_history = model.history(train_dl)
        self._val_history = model.history(val_dl)

    @staticmethod
    def _merge_histories(train: _History, val: _History, epoch: int) -> Dict:
        t = {f'train_{k}': v[epoch] for k, v in train.history.items()}
        v = {f'val_{k}': v[epoch] for k, v in val.history.items()}
        r = t.copy()
        r.update(v)
        return r

    @staticmethod
    def _train_step(bar: tqdm, model: _Model, device: torch.device, train_dl: data.DataLoader,
                    loss_fn: _LossBase, optimizer: _Optimizer, history: _History, step_scheduler):
        model.train()
        for i, ds in enumerate(train_dl):
            history.start_step()
            # All data-sets to the GPU if available
            ds = [d.to(device, non_blocking=True) for d in ds]
            optimizer.zero_grad()
            x = Trainer._get_x(model, ds)
            y = Trainer._get_y(model, ds)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            history.end_step(out, y, loss)
            if step_scheduler is not None:
                step_scheduler.step()
            del ds
            del loss
            if bar is not None:
                bar.update(1)
            if history.early_break():
                # print(f'Validation step ended early')
                break

    @staticmethod
    def _validation_step(bar: tqdm, model: _Model, device: torch.device, val_ds: data.DataLoader,
                         loss_fn: _LossBase, history: _History):
        with torch.no_grad():
            model.eval()
            for i, ds in enumerate(val_ds):
                history.start_step()
                # All data-sets to the GPU if available
                ds = [d.to(device, non_blocking=True) for d in ds]
                x = Trainer._get_x(model, ds)
                y = Trainer._get_y(model, ds)
                out = model(x)
                loss = loss_fn(out, y)
                history.end_step(out, y, loss)
                del ds
                del loss
                if bar is not None:
                    bar.update(1)
                if history.early_break():
                    # print(f'Validation step ended early')
                    break

    def _train(self, epochs: int, loss_fn: _LossBase, o: _Optimizer, step_scheduler) -> Tuple[_History, _History]:
        self._model.to(self._device)
        for epoch in range(epochs):
            self._train_history.start_epoch()
            self._val_history.start_epoch()
            with tqdm(total=self._train_history.steps+self._val_history.steps,
                      desc=f'Epoch {epoch+1:03d}/{epochs:03d}') as bar:
                Trainer._train_step(bar, self._model, self._device, self._train_dl, loss_fn, o,
                                    self._train_history, step_scheduler)
                Trainer._validation_step(bar, self._model, self._device, self._val_dl, loss_fn, self._val_history)
                self._train_history.end_epoch()
                self._val_history.end_epoch()
                bar.set_postfix(self._merge_histories(self._train_history, self._val_history, epoch))
        return self._train_history, self._val_history

    def find_lr(self, start_lr: float, end_lr: float, max_steps: int = 100, wd: float = 0.0) -> LRHistory:
        # Save model and optimizer state, so we can restore
        save_file = './temp_model.pt'
        logger.info(f'Saving model under {save_file}')
        if os.path.exists(save_file):
            os.remove(save_file)
        self.save(save_file)
        self._model.to(self._device)
        # Set-up a step schedule with new optimizer. It will adjust (increase) the LR at each step.
        o = self.model.optimizer(start_lr, wd)
        lr_schedule = LinearLR(end_lr, max_steps, o)
        history = LRHistory(self._train_dl, lr_schedule, 5, 0.1, max_steps)
        # Run Loop
        with tqdm(total=min(max_steps, history.steps), desc=f'Finding LR in {max_steps} steps') as bar:
            self._train_step(bar, self._model, self.device, self._train_dl, self.model.loss_fn, o,
                             history, lr_schedule)
        # Restore model and optimizer
        logger.info(f'Restoring model from {save_file}')
        self.load(save_file)
        os.remove(save_file)
        return history

    def train_one_cycle(self, epochs: int, max_lr: float, wd: float = None, pct_start: float = 0.3,
                        div_factor: float = 25, final_div_factor: float = 1e4) -> Tuple[_History, _History]:
        """Train a model with the one cycle policy as explained by Leslie Smith; https://arxiv.org/pdf/1803.09820.pdf.
        One cycle training normally has faster convergence. It basically first increases the learning rate to a
        maximum specified rate and then gradually decreases it to a minimum.

        :param epochs: Number of epoch to train for.
        :param max_lr: Maximum learning rate to use.
        :param wd: The weight decay. Default depends on the model.
        :param pct_start: The percentage of the cycle during which the learning rate is increased. Default = 0.3
        :param div_factor: Defines the initial learning rate as max_lr/div_factor. Default = 25
        :param final_div_factor: Defined the final learning rate as initial_lr/final_div_factor. Default = 0.0001
        :return: 2 Objects of type LR_History. The contain the training statistics for the training and validation steps
            respectively.
        """
        o = self.model.optimizer(max_lr, wd)
        # noinspection PyUnresolvedReferences
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            o.optimizer,
            max_lr=max_lr,
            steps_per_epoch=self._train_history.steps,
            epochs=epochs,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            pct_start=pct_start
        )
        # inspection PyUnresolvedReferences
        return self._train(epochs, self.model.loss_fn, o, scheduler)


class Tester(_ModelManager):
    """Class to test a Neural net. Embeds some methods that hide the Pytorch logic.

    :argument model: The model to be tested. This needs to be a d373c7 model. Not a regular nn.Module.
    :argument device: A torch device (CPU or GPU) to use during training.
    :argument test_dl: A torch DataLoader object containing the test data
    """
    def __init__(self, model: _Model, device: torch.device, test_dl: data.DataLoader):
        _ModelManager.__init__(self, model, device)
        self._test_dl = test_dl

    @staticmethod
    def _test_step(model: _Model, device: torch.device, test_dl: data.DataLoader) -> List[torch.Tensor]:
        model.eval()
        out = []
        with torch.no_grad():
            with tqdm(total=len(test_dl), desc=f'Testing in {len(test_dl)} steps') as bar:
                for i, ds in enumerate(test_dl):
                    # All data-sets to the GPU if available
                    ds = [d.to(device, non_blocking=True) for d in ds]
                    x = Tester._get_x(model, ds)
                    out.append((model(x)))
                    bar.update(1)
                    del ds
        return out

    def test_plot(self) -> Tuple[np.array, np.array]:
        # Run test and convert to numpy Array for easy of use in plotting routines
        self._model.to(self._device)
        pr = Tester._test_step(self._model, self._device, self._test_dl)
        pr = torch.cat(pr, dim=0).cpu().numpy()
        lb = [Tester._get_y(self._model, ds)[0] for ds in iter(self._test_dl)]
        lb = torch.squeeze(torch.cat(lb, dim=0)).cpu().numpy()
        return pr, lb

    @staticmethod
    def _score_step(model: _Model, device: torch.device, test_dl: data.DataLoader,
                    loss_fn: _LossBase) -> List[torch.Tensor]:
        model.eval()
        score = []
        # Start loop
        with torch.no_grad():
            with tqdm(total=len(test_dl), desc=f'Creating Scores in {len(test_dl)} steps') as bar:
                for i, ds in enumerate(test_dl):
                    # All data-sets to the GPU
                    ds = [d.to(device, non_blocking=True) for d in ds]
                    x = Tester._get_x(model, ds)
                    y = Trainer._get_y(model, ds)
                    s = loss_fn.score(model(x), y)
                    score.append(s)
                    bar.update(1)
                    del ds
        return score

    def score_plot(self) -> Tuple[np.array, np.array]:
        # Model to GPU
        label_index = self.model.label_index
        self.model.to(self._device)
        scores = Tester._score_step(self.model, self._device, self._test_dl, self.model.loss_fn)
        scores = torch.cat(scores, dim=0).cpu().numpy()
        lb = [ds[label_index] for ds in iter(self._test_dl)]
        lb = torch.squeeze(torch.cat(lb, dim=0)).cpu().numpy()
        return scores, lb
