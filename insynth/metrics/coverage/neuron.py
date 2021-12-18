import random
from copy import deepcopy

import numpy as np

from insynth.metrics.coverage.coverage_calculator import AbstractCoverageCalculator


def num_neurons(shape):
    return np.prod([dim for dim in shape if dim is not None])


def _init_dict(model, initial_value) -> dict:
    coverage_dict = {}
    for layer in get_layers_with_neurons(model):
        coverage_dict[layer.name] = np.full((num_neurons(layer.output_shape)), initial_value)
    return coverage_dict


def get_layers_with_neurons(model):
    return [layer for layer in model.layers if
            'flatten' not in layer.name and 'input' not in layer.name and 'embedding' not in layer.name and 'dropout' not in layer.name]


def get_model_activations(model, input_data):
    from tensorflow import keras
    layers = get_layers_with_neurons(model)
    intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                  outputs=[layer.output for layer in
                                                           layers])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data, verbose=0)
    return intermediate_layer_outputs


def neurons_covered(coverage_dict):
    covered_neurons = sum(arr.sum() for arr in coverage_dict.values())
    total_neurons = sum(len(arr) for arr in coverage_dict.values())
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def get_random_uncovered_neuron(coverage_dict):
    uncovered_neurons = []
    for layer_name, arr in coverage_dict.items():
        uncovered_neurons_indices = np.argwhere(arr == False).flatten()
        uncovered_neurons.extend([(layer_name, neuron_index) for neuron_index in uncovered_neurons_indices])
    if uncovered_neurons:
        return random.choice(uncovered_neurons)
    else:
        layer_name = random.choice(list(coverage_dict.keys()))
        neuron_index = random.choice(coverage_dict[layer_name])
        return (layer_name, neuron_index)


def iterate_over_layer_activations(model, layers, input_data):
    layer_names = [layer.name for layer in layers]
    intermediate_layer_activations = get_model_activations(model, input_data)
    return zip(layer_names, map(lambda x: x[0], intermediate_layer_activations))


def iterate_over_neuron_activations(layer_activations):
    activations_shape = layer_activations.shape
    neuron_indices = range(num_neurons(activations_shape))
    return zip(neuron_indices, layer_activations.flatten())


def merge_np_arrays(arr1, arr2):
    arr1[arr2] = True
    return arr1


def merge_dicts(dict_1, dict_2):
    for key in dict_1.keys():
        dict_1[key] = merge_np_arrays(dict_1[key], dict_2[key])
    return dict_1


class NeuronCoverageCalculator(AbstractCoverageCalculator):
    def __copy__(self):
        self_copy = NeuronCoverageCalculator(self.model, self.activation_threshold)
        self_copy._layers_with_neurons = deepcopy(self._layers_with_neurons)
        self_copy.coverage_dict = deepcopy(self.coverage_dict)
        return self_copy

    def __init__(self, model, activation_threshold=0):
        super().__init__(model)
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.activation_threshold = activation_threshold
        self.coverage_dict = _init_dict(model, False)

    def update_coverage(self, input_data):
        for layer_name, layer_activations in iterate_over_layer_activations(self.model, self._layers_with_neurons,
                                                                            input_data):
            layer_coverage_arr = self.coverage_dict[layer_name]
            layer_activations = layer_activations.flatten()
            layer_coverage_arr[layer_activations > self.activation_threshold] = True

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = neurons_covered(self.coverage_dict)
        return {
            'total_neurons': total_neurons,
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage
        }

    def merge(self, other_calculator):
        self.coverage_dict = merge_dicts(self.coverage_dict, other_calculator.coverage_dict)


class StrongNeuronActivationCoverageCalculator(AbstractCoverageCalculator):
    def __copy__(self):
        self_copy = StrongNeuronActivationCoverageCalculator(self.model)
        self_copy._layers_with_neurons = deepcopy(self._layers_with_neurons)
        self_copy.coverage_dict = deepcopy(self.coverage_dict)
        self_copy.upper_neuron_bounds_dict = deepcopy(self.upper_neuron_bounds_dict)
        self_copy.lower_neuron_bounds_dict = deepcopy(self.lower_neuron_bounds_dict)
        return self_copy

    def __init__(self, model):
        super().__init__(model)
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.coverage_dict = _init_dict(model, False)
        self.upper_neuron_bounds_dict = _init_dict(model, np.NAN)
        self.lower_neuron_bounds_dict = _init_dict(model, np.NAN)

    def update_neuron_bounds(self, input_data):
        for layer_name, layer_activations in iterate_over_layer_activations(self.model, self._layers_with_neurons,
                                                                            input_data):
            upper_neuron_bounds_dict = self.upper_neuron_bounds_dict[layer_name]
            lower_neuron_bounds_dict = self.lower_neuron_bounds_dict[layer_name]
            layer_activations = layer_activations.flatten()

            upper_neuron_bounds_dict[
                (layer_activations > upper_neuron_bounds_dict) | np.isnan(upper_neuron_bounds_dict)] = \
                layer_activations[
                    (layer_activations > upper_neuron_bounds_dict) | np.isnan(upper_neuron_bounds_dict)]

            lower_neuron_bounds_dict[
                (layer_activations < lower_neuron_bounds_dict) | np.isnan(lower_neuron_bounds_dict)] = \
                layer_activations[
                    (layer_activations < lower_neuron_bounds_dict) | np.isnan(lower_neuron_bounds_dict)]

    def update_coverage(self, input_data):
        for layer_name, layer_activations in iterate_over_layer_activations(self.model, self._layers_with_neurons,
                                                                            input_data):
            layer_coverage_dict = self.coverage_dict[layer_name]
            upper_neuron_bounds_dict = self.upper_neuron_bounds_dict[layer_name]
            layer_activations = layer_activations.flatten()
            layer_coverage_dict[layer_activations > upper_neuron_bounds_dict] = True

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = neurons_covered(self.coverage_dict)
        return {
            'total_neurons': total_neurons,
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage
        }

    def merge(self, other_calculator):
        self.coverage_dict = merge_dicts(self.coverage_dict, other_calculator.coverage_dict)


class KMultiSectionNeuronCoverageCalculator(StrongNeuronActivationCoverageCalculator):
    def __copy__(self):
        self_copy = KMultiSectionNeuronCoverageCalculator(self.model, self.k)
        self_copy._layers_with_neurons = deepcopy(self._layers_with_neurons)
        self_copy.coverage_dict = deepcopy(self.coverage_dict)
        self_copy.upper_neuron_bounds_dict = deepcopy(self.upper_neuron_bounds_dict)
        self_copy.lower_neuron_bounds_dict = deepcopy(self.lower_neuron_bounds_dict)
        return self_copy

    def __init__(self, model, k=3):
        super().__init__(model)
        self.k = k
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.upper_neuron_bounds_dict = _init_dict(model, np.NAN)
        self.lower_neuron_bounds_dict = _init_dict(model, np.NAN)
        self.coverage_dict = self._init_dict(model)

    def update_coverage(self, input_data):
        for layer_name, layer_activations in iterate_over_layer_activations(self.model, self._layers_with_neurons,
                                                                            input_data):
            layer_coverage_dict = self.coverage_dict[layer_name]
            upper_neuron_bounds_arr = self.upper_neuron_bounds_dict[layer_name]
            lower_neuron_bounds_arr = self.lower_neuron_bounds_dict[layer_name]
            layer_activations = layer_activations.flatten()

            step_sizes = (upper_neuron_bounds_arr - lower_neuron_bounds_arr) / self.k
            activated_sections = ((layer_activations - lower_neuron_bounds_arr) / step_sizes).astype(int)

            layer_coverage_dict[(0 <= activated_sections) & (activated_sections < self.k), activated_sections[
                (0 <= activated_sections) & (activated_sections < self.k)]] = True

    def _init_dict(self, model):
        coverage_dict = {}
        for layer in get_layers_with_neurons(model):
            layer_name = layer.name
            coverage_dict[layer_name] = np.full((num_neurons(layer.output_shape), self.k), False)
        return coverage_dict

    def neurons_covered(self):
        covered_neurons = sum((arr.sum(axis=1) == self.k).sum() for arr in self.coverage_dict.values())
        total_neurons = sum(len(arr) for arr in self.coverage_dict.values())
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def sections_covered(self):
        covered_sections = sum(arr.sum() for arr in self.coverage_dict.values())
        total_sections = sum(len(arr) for arr in self.coverage_dict.values()) * self.k
        return covered_sections, total_sections, covered_sections / float(total_sections)

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = self.neurons_covered()
        covered_sections, total_sections, sections_covered_percentage = self.sections_covered()
        return {
            'total_neurons': total_neurons,
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage,
            'total_sections': total_sections,
            'covered_sections': covered_sections,
            'sections_covered_percentage': sections_covered_percentage,
        }

    def merge(self, other_calculator):
        self.coverage_dict = merge_dicts(self.coverage_dict, other_calculator.coverage_dict)


class NeuronBoundaryCoverageCalculator(StrongNeuronActivationCoverageCalculator):
    def __copy__(self):
        self_copy = NeuronBoundaryCoverageCalculator(self.model)
        self_copy._layers_with_neurons = deepcopy(self._layers_with_neurons)
        self_copy.coverage_dict = deepcopy(self.coverage_dict)
        self_copy.upper_neuron_bounds_dict = deepcopy(self.upper_neuron_bounds_dict)
        self_copy.lower_neuron_bounds_dict = deepcopy(self.lower_neuron_bounds_dict)
        return self_copy

    def __init__(self, model):
        super().__init__(model)
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.coverage_dict = self._init_dict(model)
        self.upper_neuron_bounds_dict = _init_dict(model, np.NAN)
        self.lower_neuron_bounds_dict = _init_dict(model, np.NAN)

    def update_coverage(self, input_data):
        for layer_name, layer_activations in iterate_over_layer_activations(self.model, self._layers_with_neurons,
                                                                            input_data):
            layer_coverage_dict = self.coverage_dict[layer_name]
            upper_neuron_bounds_arr = self.upper_neuron_bounds_dict[layer_name]
            lower_neuron_bounds_arr = self.lower_neuron_bounds_dict[layer_name]
            layer_activations = layer_activations.flatten()

            layer_coverage_dict[layer_activations > upper_neuron_bounds_arr, 1] = True
            layer_coverage_dict[layer_activations < lower_neuron_bounds_arr, 0] = True

    def neurons_covered(self):
        covered_neurons = sum((arr.sum(axis=1) == 2).sum() for arr in self.coverage_dict.values())
        total_neurons = sum(len(arr) for arr in self.coverage_dict.values())
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def corners_covered(self):
        covered_corners = sum(arr.sum() for arr in self.coverage_dict.values())
        total_corners = sum(len(arr) for arr in self.coverage_dict.values()) * 2
        return covered_corners, total_corners, covered_corners / float(total_corners)

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = self.neurons_covered()
        covered_corners, total_corners, corners_covered_percentage = self.corners_covered()
        return {
            'total_neurons': total_neurons,
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage,
            'total_corners': total_corners,
            'covered_corners': covered_corners,
            'corners_covered_percentage': corners_covered_percentage,
        }

    def _init_dict(self, model):
        coverage_dict = {}
        for layer in get_layers_with_neurons(model):
            layer_name = layer.name
            coverage_dict[layer_name] = np.full((num_neurons(layer.output_shape), 2), False)
        return coverage_dict

    def merge(self, other_calculator):
        self.coverage_dict = merge_dicts(self.coverage_dict, other_calculator.coverage_dict)


class TopKNeuronCoverageCalculator(AbstractCoverageCalculator):
    def __copy__(self):
        self_copy = TopKNeuronCoverageCalculator(self.model, self.k)
        self_copy._layers_with_neurons = deepcopy(self._layers_with_neurons)
        self_copy.coverage_dict = deepcopy(self.coverage_dict)
        self.k = deepcopy(self.k)
        return self_copy

    def get_random_uncovered_neuron(self):
        uncovered_neurons = []
        for layer in get_layers_with_neurons(self.model):
            for neuron_index in range(num_neurons(layer.output_shape)):
                if neuron_index not in self.coverage_dict[layer.name]:
                    uncovered_neurons.append((layer.name, neuron_index))
        if uncovered_neurons:
            return random.choice(uncovered_neurons)
        else:
            return None

    def __init__(self, model, k=3):
        super().__init__(model)
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.coverage_dict = self._init_dict(model)
        self.k = k

    def _init_dict(self, model) -> dict:
        coverage_dict = {}
        for layer in get_layers_with_neurons(model):
            coverage_dict[layer.name] = set()
        return coverage_dict

    def merge(self, other_calculator):
        for key in self.coverage_dict.keys():
            self.coverage_dict[key] |= other_calculator.coverage_dict[key]

    def update_coverage(self, input_data):

        for layer_name, layer_activations in iterate_over_layer_activations(self.model, self._layers_with_neurons,
                                                                            input_data):
            coverage_dict = self.coverage_dict[layer_name]
            layer_activations = layer_activations.flatten()
            k = min(len(layer_activations), self.k)
            top_k_indices = np.argpartition(layer_activations, -k)[-k:]
            coverage_dict |= set(top_k_indices)

    def get_coverage(self) -> dict:
        top_k_neurons = sum(len(layer) for layer in self.coverage_dict.values())
        total_neurons = sum(num_neurons(layer.output_shape) for layer in get_layers_with_neurons(self.model))
        return {
            'total_neurons': total_neurons,
            'top_k_neurons': top_k_neurons,
            'top_k_neuron_coverage_percentage': top_k_neurons / total_neurons
        }


class TopKNeuronPatternsCalculator(AbstractCoverageCalculator):
    def merge(self, other_calculator):
        self.coverage_dict |= other_calculator.coverage_dict

    def __copy__(self):
        self_copy = TopKNeuronPatternsCalculator(self.model, self.k)
        self_copy._layers_with_neurons = deepcopy(self._layers_with_neurons)
        self_copy.coverage_dict = deepcopy(self.coverage_dict)
        self.k = deepcopy(self.k)
        return self_copy

    def get_random_uncovered_neuron(self):
        raise NotImplementedError

    def __init__(self, model, k=3):
        super().__init__(model)
        self.k = k
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.coverage_dict = self._init_dict()

    def _init_dict(self) -> set:
        coverage_dict = set()
        return coverage_dict

    def update_coverage(self, input_data):
        pattern = []

        for layer_name, layer_activations in iterate_over_layer_activations(self.model, self._layers_with_neurons,
                                                                            input_data):
            layer_activations = layer_activations.flatten()
            top_k_indices = (-layer_activations).argsort()[:self.k]
            pattern.extend(map(lambda index: layer_name + '_' + str(index), top_k_indices))

        self.coverage_dict |= {tuple(pattern)}

    def get_coverage(self) -> dict:
        return {
            'total_patterns': len(self.coverage_dict),
        }
