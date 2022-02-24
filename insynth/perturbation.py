#  Copyright (c) 2022, Chair of Software Technology
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#  - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  - Neither the name of the University Mannheim nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import random
from abc import ABC, abstractmethod

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
            return original_input
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
