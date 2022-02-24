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
from copy import deepcopy

import numpy as np

from insynth.calculators.abstract_calculator import AbstractCoverageCalculator, num_neurons


class TopKNeuronPatternsCalculator(AbstractCoverageCalculator):
    def merge(self, other_calculator):
        self.coverage_dict |= other_calculator.coverage_dict

    def __copy__(self):
        self_copy = TopKNeuronPatternsCalculator(self.model, self.k)
        self_copy.coverage_dict = deepcopy(self.coverage_dict)
        return self_copy

    def __init__(self, model, k=3):
        super().__init__(model)
        self.k = k
        self.coverage_dict = self._init_dict()

    def _init_dict(self) -> set:
        coverage_dict = set()
        return coverage_dict

    def update_coverage(self, input_data):
        pattern = []

        for layer_name, layer_activations in self.iterate_over_layer_activations(
                input_data):
            layer_activations = layer_activations.flatten()
            top_k_indices = (-layer_activations).argsort()[:self.k]
            pattern.extend(map(lambda index: layer_name + '_' + str(index), top_k_indices))

        self.coverage_dict |= {tuple(pattern)}

    def get_coverage(self) -> dict:
        return {
            'total_patterns': len(self.coverage_dict),
        }


class TopKNeuronCoverageCalculator(AbstractCoverageCalculator):
    def __copy__(self):
        self_copy = TopKNeuronCoverageCalculator(self.model, self.k)
        self_copy.coverage_dict = deepcopy(self.coverage_dict)
        return self_copy

    def __init__(self, model, k=3):
        super().__init__(model)
        self.coverage_dict = self._init_dict(model)
        self.k = k

    def _init_dict(self, model) -> dict:
        coverage_dict = {}
        for layer in self.layers_with_neurons:
            coverage_dict[layer.name] = set()
        return coverage_dict

    def merge(self, other_calculator):
        for key in self.coverage_dict.keys():
            self.coverage_dict[key] |= other_calculator.coverage_dict[key]

    def update_coverage(self, input_data):

        for layer_name, layer_activations in self.iterate_over_layer_activations(
                input_data):
            coverage_dict = self.coverage_dict[layer_name]
            layer_activations = layer_activations.flatten()
            k = min(len(layer_activations), self.k)
            top_k_indices = np.argpartition(layer_activations, -k)[-k:]
            coverage_dict |= set(top_k_indices)

    def get_coverage(self) -> dict:
        top_k_neurons = sum(len(layer) for layer in self.coverage_dict.values())
        total_neurons = sum(num_neurons(layer) for layer in self.layers_with_neurons)
        return {
            'total_neurons': total_neurons,
            'top_k_neurons': top_k_neurons,
            'top_k_neuron_coverage_percentage': top_k_neurons / total_neurons
        }