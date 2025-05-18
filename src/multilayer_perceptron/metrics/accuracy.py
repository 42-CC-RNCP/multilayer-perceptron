import numpy as np
from litetorch.core.tensor import Tensor
from sklite.metrics import AccuracyScore
from .base import Metric


class TrainAccuracy(Metric):
    def __call__(self, trainer):
        y_true, y_pred = np.array([]), np.array([])
        for X_batch, y_batch in trainer.train_loader:
            X_batch: Tensor
            y_batch: Tensor
            probs = trainer.model(X_batch).data.ravel()
            preds = (probs >= 0.5).astype(int)
            y_true = np.concatenate((y_true, y_batch.data.ravel()))
            y_pred = np.concatenate((y_pred, preds))
    
        return AccuracyScore()(y_true, y_pred)
    

class ValAccuracy(Metric):
    def __call__(self, trainer):
        y_true, y_pred = np.array([]), np.array([])
        for X_batch, y_batch in trainer.val_loader:
            X_batch: Tensor
            y_batch: Tensor
            probs = trainer.model(X_batch).data.ravel()
            preds = (probs >= 0.5).astype(int)
            y_true = np.concatenate((y_true, y_batch.data.ravel()))
            y_pred = np.concatenate((y_pred, preds))
            
        return AccuracyScore()(y_true, y_pred)
            