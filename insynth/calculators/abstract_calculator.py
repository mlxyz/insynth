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

from abc import ABC, abstractmethod

import numpy as np
from tensorflow import keras


def num_neurons(layer):
    """
    Returns the number of neurons in a layer.
    :param layer:
    :return:
    """
    return np.prod([dim for dim in layer.output_shape if dim is not None])


class AbstractCoverageCalculator(ABC):
    def __init__(self, model):
        self.model = model
        self.layers_with_neurons = self.get_layers_with_neurons()
        self.intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                           outputs=[layer.output for layer in
                                                                    self.layers_with_neurons])

    def get_layers_with_neurons(self):
        """
        Returns a list of all layers that have at least one neuron.
        :return:
        """
        return [layer for layer in self.model.layers if
                'flatten' not in layer.name and 'input' not in layer.name and 'embedding' not in layer.name and 'dropout' not in layer.name]

    def get_model_activations(self, input_data):
        """
        Returns the activations of all neurons for the given input data.
        :param input_data:
        :return:
        """
        intermediate_layer_outputs = [tensor.numpy() for tensor in
                                      self.intermediate_layer_model(input_data, training=False)]
        return intermediate_layer_outputs

    def _init_plain_coverage_dict(self, initial_value) -> dict:
        """
        Initializes a coverage dictionary for this model with the given initial value.
        :param initial_value:
        :return:
        """
        coverage_dict = {}
        for layer in self.layers_with_neurons:
            coverage_dict[layer.name] = np.full((num_neurons(layer)),
                                                initial_value)
        return coverage_dict

    def iterate_over_layer_activations(self, input_data):
        """
        Iterates over all layer activations and yields the layer name and the activations of its neurons.
        :param input_data:
        :return:
        """
        layer_names = [layer.name for layer in self.layers_with_neurons]
        intermediate_layer_activations = self.get_model_activations(input_data)
        return zip(layer_names, map(lambda x: x[0], intermediate_layer_activations))

    @abstractmethod
    def update_coverage(self, input_data) -> dict:
        """
        Updates the coverage dictionary for the given input data.
        :param input_data:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def __copy__(self):
        raise NotImplementedError

    @abstractmethod
    def merge(self, other_calculator):
        """
        Merges the coverage dictionaries of this calculator with the coverage dictionaries of the given calculator.
        This yields a new calculator that contains the  coverage dictionaries that would result from creating a new calculator and passing the data from this and other_calculator to it individually.
        :param other_calculator:
        :return:
        """
        raise NotImplementedError
