from abc import ABC, abstractmethod

import numpy as np
from tensorflow import keras

def num_neurons(layer):
    return np.prod([dim for dim in layer.output_shape if dim is not None])
class AbstractCoverageCalculator(ABC):
    def __init__(self, model):
        self.model = model
        self.layers_with_neurons = self.get_layers_with_neurons()
        self.intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                           outputs=[layer.output for layer in
                                                                    self.layers_with_neurons])

    def get_layers_with_neurons(self):
        return [layer for layer in self.model.layers if
                'flatten' not in layer.name and 'input' not in layer.name and 'embedding' not in layer.name and 'dropout' not in layer.name]

    def get_model_activations(self, input_data):
        intermediate_layer_outputs = [tensor.numpy() for tensor in
                                      self.intermediate_layer_model(input_data, training=False)]
        return intermediate_layer_outputs

    def _init_plain_coverage_dict(self, initial_value) -> dict:
        coverage_dict = {}
        for layer in self.layers_with_neurons:
            coverage_dict[layer.name] = np.full((num_neurons(layer)),
                                                initial_value)
        return coverage_dict

    def iterate_over_layer_activations(self, input_data):
        layer_names = [layer.name for layer in self.layers_with_neurons]
        intermediate_layer_activations = self.get_model_activations(input_data)
        return zip(layer_names, map(lambda x: x[0], intermediate_layer_activations))

    @abstractmethod
    def update_coverage(self, input_data) -> dict:
        raise NotImplementedError

    @abstractmethod
    def __copy__(self):
        raise NotImplementedError

    @abstractmethod
    def merge(self, other_calculator):
        raise NotImplementedError
