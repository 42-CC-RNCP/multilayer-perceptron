from litetorch.core.tensor import Tensor
from .base import Metric


class TrainLoss(Metric):
    def __call__(self, trainer):
        return trainer.train_losses[-1] if trainer.train_losses else 0.0
    
    def __str__(self):
        return "TrainLoss"


class ValLoss(Metric):
    def __call__(self, trainer):
        return trainer.val_losses[-1] if trainer.val_losses else 0.0
    
    def __str__(self):
        return "ValLoss"
