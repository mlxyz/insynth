import unittest

import numpy as np
from tensorflow import keras

from insynth.metrics.coverage.neuron import NeuronCoverageCalculator


class TestNeuronCoverageCalculator(unittest.TestCase):

    def _generate_simple_feedforward_model(self):
        model = keras.Sequential()

        layer = keras.layers.Dense(2, input_shape=(2,), activation='linear')
        model.add(layer)
        weights = np.array([[0, 1],
                            [1, 0]])
        layer.set_weights([weights, np.zeros((2,))])

        layer = keras.layers.Dense(2, activation='linear')
        model.add(layer)

        weights = np.array([[1, 1],
                            [1, 1]])
        layer.set_weights([weights, np.zeros((2,))])

        layer = keras.layers.Dense(1, activation='sigmoid')
        model.add(layer)
        weights = np.array([[0],
                            [1]])
        layer.set_weights([weights, np.zeros((1,))])

        return model

    def test_NeuronCoverageCalculator(self):
        model = self._generate_simple_feedforward_model()
        calc = NeuronCoverageCalculator(model, activation_threshold=0.8)

        calc.update_coverage(np.array([[0, 1]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert coverage['total_neurons'] == 5
        assert coverage['covered_neurons'] == 3
        assert coverage['covered_neurons_percentage'] == 3 / 5
        assert (uncovered_neuron == ('dense_2', 0)) or (uncovered_neuron == ('dense', 1))

        calc.update_coverage(np.array([[1, 0]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert coverage['total_neurons'] == 5
        assert coverage['covered_neurons'] == 4
        assert uncovered_neuron == ('dense_2', 0)

        calc.update_coverage(np.array([[1, 1]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert coverage['total_neurons'] == 5
        assert coverage['covered_neurons'] == 5
        assert uncovered_neuron is not None
