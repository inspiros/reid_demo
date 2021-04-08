__all__ = ['IncrementalMeanTracker']


class IncrementalMeanTracker:
    def __init__(self):
        self.sum = self.mean = self.count = 0

    def update(self, val):
        self.count += 1
        if self.count == 1:
            self.sum = self.mean = val
        else:
            self.sum += val
            self.mean = self.sum / self.count
        return self.mean
