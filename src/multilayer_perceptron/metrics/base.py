from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """
    Abstract base class for all metrics.
    """

    @abstractmethod
    def __call__(self, trainer):
        pass
    
    def __str__(self):
        return self.__class__.__name__
