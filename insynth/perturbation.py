from abc import ABC, abstractmethod
import random

import numpy as np

from insynth.metrics.coverage.neuron import NeuronCoverageCalculator, StrongNeuronActivationCoverageCalculator, \
    NeuronBoundaryCoverageCalculator, KMultiSectionNeuronCoverageCalculator, TopKNeuronCoverageCalculator, \
    TopKNeuronPatternsCalculator


class AbstractBlackboxPerturbator(ABC):
    def __init__(self, p=0.5):
        self.p = p

    def apply(self, original_input):
        if random.random() > self.p:
            return original_input
        return self._internal_apply(original_input)

    @abstractmethod
    def _internal_apply(self, original_input):
        raise NotImplementedError


class BlackboxImagePerturbator(AbstractBlackboxPerturbator):
    def __init__(self, p=0.5):
        super().__init__(p)

    @abstractmethod
    def _internal_apply(self, original_input):
        raise NotImplementedError


class BlackboxAudioPerturbator(AbstractBlackboxPerturbator):
    def __init__(self, p=0.5):
        super().__init__(p)

    def apply(self, original_input):
        if random.random() > self.p:
            return original_input[1]
        return self._internal_apply(original_input)

    @abstractmethod
    def _internal_apply(self, original_input):
        raise NotImplementedError


class BlackboxTextPerturbator(AbstractBlackboxPerturbator):
    def __init__(self, p=0.5):
        super().__init__(p)

    @abstractmethod
    def _internal_apply(self, original_input):
        raise NotImplementedError


class AbstractWhiteboxPerturbator(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def apply(self, original_input):
        raise NotImplementedError


class WhiteboxImagePerturbator(AbstractWhiteboxPerturbator):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def apply(self, original_input):
        raise NotImplementedError


class WhiteboxAudioPerturbator(AbstractWhiteboxPerturbator):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def apply(self, original_input):
        raise NotImplementedError


class WhiteboxTextPerturbator(AbstractWhiteboxPerturbator):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def apply(self, original_input):
        raise NotImplementedError


COVERAGE_CRITERIA_TO_CALCULATOR_CLASS = {
    'NC': NeuronCoverageCalculator,
    'SNAC': StrongNeuronActivationCoverageCalculator,
    'NBC': NeuronBoundaryCoverageCalculator,
    'KMSNC': KMultiSectionNeuronCoverageCalculator,
    'TKNC': TopKNeuronCoverageCalculator,
    'TKPC': TopKNeuronPatternsCalculator
}

