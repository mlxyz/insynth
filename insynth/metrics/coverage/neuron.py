from collections import defaultdict

import numpy as np

from insynth.metrics.coverage.coverage_calculator import AbstractCoverageCalculator
from tensorflow import keras


class NeuronCoverageCalculator(AbstractCoverageCalculator):
    def __init__(self, model, activation_threshold):
        super().__init__(model)
        self._layers_with_neurons = self._get_layers_with_neurons()
        self.activation_threshold = activation_threshold
        self.coverage_dict = self._init_dict()

    def _get_layers_with_neurons(self):
        return [layer for layer in self.model.layers if
                'flatten' not in layer.name and 'input' not in layer.name]

    def update_coverage(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_model = keras.models.Model(inputs=self.model.input,
                                                      outputs=[layer.output for layer in
                                                               layers])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_outputs):
            layer_activations = intermediate_layer_output[0]
            activations_shape = layer_activations.shape
            for neuron_index in range(NeuronCoverageCalculator._num_neurons(activations_shape)):
                if layer_activations[np.unravel_index(neuron_index, activations_shape)] > self.activation_threshold:
                    self.coverage_dict[(layer_name, neuron_index)] = True

    @staticmethod
    def _num_neurons(shape):
        return np.prod([dim for dim in shape if dim is not None])

    def _init_dict(self) -> dict:
        coverage_dict = defaultdict(bool)
        for layer in [layer for layer in self._layers_with_neurons]:
            for index in range(NeuronCoverageCalculator._num_neurons(layer.output_shape)):  # product of dims
                coverage_dict[(layer.name, index)] = False
        return coverage_dict

    def _neuron_covered(self):
        covered_neurons = sum(neuron for neuron in self.coverage_dict.values() if neuron)
        total_neurons = len(self.coverage_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = self._neuron_covered()
        return {
            'total_neurons': len(self.coverage_dict),
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage
        }
