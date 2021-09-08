from abc import ABC, abstractmethod


class AbstractCoverageCalculator(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def update_coverage(self, input_data) -> dict:
        pass
