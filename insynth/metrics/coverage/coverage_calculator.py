from abc import ABC, abstractmethod


class AbstractCoverageCalculator(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def update_coverage(self, input_data) -> dict:
        pass

    @abstractmethod
    def get_random_uncovered_neuron(self):
        pass
