import torch
from typing import Optional
from numpy.typing import ArrayLike

from pytorcher.metrics import Metric

class Mean(Metric):
    """
    Mean Metric.

    This metric computes the mean of the values provided to it.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "mean")

    def update_state(self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike] = None):

        batch_total = torch.sum(y_pred).item()
        batch_count = y_pred.numel()

        self._total += batch_total
        self._count += batch_count