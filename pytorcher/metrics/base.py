import abc
from typing import Optional
from numpy.typing import ArrayLike

class Metric:
    """
    Base metric interface inspired by Keras.
    Subclasses should implement update_state(...) and result().
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._total = 0.0
        self._count = 0.0
        self.reset_states()

    def reset_states(self):
        """Reset internal accumulators."""
        self._total = 0.0
        self._count = 0.0

    @abc.abstractmethod
    def update_state(self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike] = None):
        """
        Update the metric state with a new batch.
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def result(self) -> float:
        """Return the current aggregated metric value."""
        if self._count == 0:
            return 0.0
        return float(self._total / self._count)