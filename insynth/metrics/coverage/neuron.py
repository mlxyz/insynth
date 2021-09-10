import random
from collections import defaultdict

import numpy as np

from insynth.metrics.coverage.coverage_calculator import AbstractCoverageCalculator
from tensorflow import keras


def num_neurons(shape):
    return np.prod([dim for dim in shape if dim is not None])


def _init_dict(model) -> dict:
    coverage_dict = defaultdict(bool)
    for layer in [layer for layer in get_layers_with_neurons(model)]:
        for index in range(num_neurons(layer.output_shape)):  # product of dims
            coverage_dict[(layer.name, index)] = False
    return coverage_dict


def get_layers_with_neurons(model):
    return [layer for layer in model.layers if
            'flatten' not in layer.name and 'input' not in layer.name]


def get_model_activations(model, input_data):
    layers = get_layers_with_neurons(model)
    intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                  outputs=[layer.output for layer in
                                                           layers])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    return intermediate_layer_outputs


def neurons_covered(coverage_dict):
    covered_neurons = sum(neuron for neuron in coverage_dict.values() if neuron)
    total_neurons = len(coverage_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def get_random_uncovered_neuron(coverage_dict):
    uncovered_neurons = [key for key, covered in coverage_dict.items() if not covered]
    if uncovered_neurons:
        return random.choice(uncovered_neurons)
    else:
        return random.choice(coverage_dict.keys())


class NeuronCoverageCalculator(AbstractCoverageCalculator):
    def get_random_uncovered_neuron(self):
        return get_random_uncovered_neuron(self.coverage_dict)

    def __init__(self, model, activation_threshold):
        super().__init__(model)
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.activation_threshold = activation_threshold
        self.coverage_dict = _init_dict(model)

    def update_coverage(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            activations_shape = layer_activations.shape
            for neuron_index in range(num_neurons(activations_shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, activations_shape)]
                if neuron_activation > self.activation_threshold:
                    self.coverage_dict[(layer_name, neuron_index)] = True

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = neurons_covered(self.coverage_dict)
        return {
            'total_neurons': len(self.coverage_dict),
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage
        }


class StrongNeuronActivationCoverageCalculator(AbstractCoverageCalculator):
    def __init__(self, model):
        super().__init__(model)
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.coverage_dict = _init_dict(model)
        self.neuron_bounds_dict = _init_dict(model)

    def get_random_uncovered_neuron(self):
        return get_random_uncovered_neuron(self.coverage_dict)

    def update_neuron_bounds(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(num_neurons(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                neuron_position = (layer_name, neuron_index)
                if not self.neuron_bounds_dict[neuron_position]:
                    self.neuron_bounds_dict[neuron_position] = (neuron_activation, neuron_activation)
                else:
                    (lower, upper) = self.neuron_bounds_dict[neuron_position]
                    if neuron_activation > upper:
                        self.neuron_bounds_dict[neuron_position] = (lower, neuron_activation)
                    elif neuron_activation < lower:
                        self.neuron_bounds_dict[neuron_position] = (neuron_activation, upper)

    def update_coverage(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(num_neurons(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                _, high = self.neuron_bounds_dict[(layer_name, neuron_index)]
                if neuron_activation > high:
                    self.coverage_dict[(layer_name, neuron_index)] = True

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = neurons_covered(self.coverage_dict)
        return {
            'total_neurons': len(self.coverage_dict),
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage
        }
