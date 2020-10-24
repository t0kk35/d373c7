"""
Helper to plot data in iPython or similar tools
(c) TSM 2020
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import PCA
# noinspection PyProtectedMember
from ..pytorch.common import _History
# inspection PyProtectedMember
from typing import Tuple


class TrainPlot:
    def __init__(self):
        self._def_style = 'ggplot'

    def plot_history(self, train: _History, val: _History, fig_size: Tuple[float] = None):
        style.use(self._def_style)
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        plt.title('Training and Validation metrics')
        plt.xlabel('Epoch')
        axis = []
        epochs = [i for i in range(1, train.epoch+1)]

        # This logic assumes the train and validate history contain the same metrics.
        for i, k in enumerate(train.history.keys()):
            if i == 0:
                ax = plt.subplot2grid((2, 1), (i, 0))
            else:
                ax = plt.subplot2grid((2, 1), (i, 0), sharex=axis[0])
            ax.plot(epochs, train.history[k], label=f'train_{k}')
            ax.plot(epochs, val.history[k], label=f'val_{k}')
            if train.history[k][0] > train.history[k][-1]:
                ax.legend(loc=1)
            else:
                ax.legend(loc=2)
            axis.append(ax)
        plt.show()

    def plot_lr(self, history: _History, fig_size: Tuple[float, float] = None):
        style.use(self._def_style)
        lr = history.history['lr']
        loss = history.history['loss']
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        plt.plot(lr, loss)
        plt.title('Loss per Learning rate')
        plt.xlabel('Learning Rate')
        plt.xscale('log')
        plt.ylabel('Loss')
        plt.show()


class TestPlot:
    def __init__(self):
        self._def_style = 'ggplot'

    @staticmethod
    def print_classification_report(predictions: np.array, labels: np.array):
        ap_score = average_precision_score(labels, predictions)
        auc_score = roc_auc_score(labels, predictions)
        predictions = (predictions > 0.5)
        cr = classification_report(labels, predictions)
        print('------------- Classification report -----------------')
        print(cr)
        print()
        print(f'auc score : {auc_score:0.4f}')
        print(f'ap score  : {ap_score:0.4f}')
        print('-----------------------------------------------------')

    @staticmethod
    def plot_confusion_matrix(predictions: np.array, labels: np.array, fig_size: Tuple[float, float] = None):
        predictions = (predictions > 0.5)
        cm = confusion_matrix(labels, predictions)
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        c_map = plt.get_cmap('Blues')
        plt.imshow(cm, interpolation='nearest', cmap=c_map)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        class_names = ['Non-Fraud', 'Fraud']
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        q = [['TN', 'FP'], ['FN', 'TP']]
        c_map_min, c_map_max = c_map(0), c_map(256)
        cut = (cm.max() + cm.min()) / 2.0
        for i in range(2):
            for j in range(2):
                color = c_map_max if cm[i, j] < cut else c_map_min
                plt.text(j, i, f'{str(q[i][j])} = {str(cm[i][j])}', color=color, ha="center", va="center")
        plt.show()

    def plot_roc_curve(self, predictions: np.array, labels: np.array, fig_size: Tuple[float, float] = None):
        style.use(self._def_style)
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        auc_score = roc_auc_score(labels, predictions)
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        plt.plot(fpr, tpr, label=f'AUC Score = {auc_score:0.4f}')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc=4)
        plt.show()
