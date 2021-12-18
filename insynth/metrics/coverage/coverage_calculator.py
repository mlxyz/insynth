from abc import ABC, abstractmethod


class AbstractCoverageCalculator(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def update_coverage(self, input_data) -> dict:
        raise NotImplementedError

    @abstractmethod
    def __copy__(self):
        raise NotImplementedError

    @abstractmethod
    def merge(self, other_calculator):
        raise NotImplementedError
