import numpy as np

__all__ = ['IncrementalMeanTracker']


class IncrementalMeanTracker:
    def __init__(self, max_count=None):
        self.max_count = np.inf if max_count is None else max_count
        self.sum = self.mean = self.count = 0

    def update(self, val):
        self.count = min(self.count + 1, self.max_count)
        if self.count == 1:
            self.sum = self.mean = val
        else:
            self.sum += val
            self.mean = self.sum / self.count
        return self.mean
