import unittest

import numpy as np
from tensorflow import keras

from insynth.metrics.coverage.neuron import NeuronCoverageCalculator, StrongNeuronActivationCoverageCalculator, \
    KMultiSectionNeuronCoverageCalculator, NeuronBoundaryCoverageCalculator, TopKNeuronCoverageCalculator, \
    TopKNeuronPatternsCalculator


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

    def tearDown(self) -> None:
        keras.backend.clear_session()

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

    def test_StrongNeuronActivationCoverageCalculator(self):
        model = self._generate_simple_feedforward_model()
        calc = StrongNeuronActivationCoverageCalculator(model)
        calc.update_neuron_bounds([np.array([[0, 1]])])

        calc.update_coverage(np.array([[1, 0]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert coverage['total_neurons'] == 5
        assert coverage['covered_neurons'] == 1
        assert uncovered_neuron != ('dense', 1)

        calc.update_coverage(np.array([[1, 1]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert coverage['total_neurons'] == 5
        assert coverage['covered_neurons'] == 4
        assert uncovered_neuron is not None

    def test_KMultiSectionNeuronCoverageCalculator(self):
        model = self._generate_simple_feedforward_model()
        calc = KMultiSectionNeuronCoverageCalculator(model)
        calc.update_neuron_bounds([np.array([[1, 1]])])
        calc.update_neuron_bounds([np.array([[0, 0]])])

        calc.update_coverage(np.array([[0.5, 0.5]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert coverage['total_sections'] == 5 * 3
        assert coverage['covered_sections'] == 5
        assert coverage['sections_covered_percentage'] == 1 / 3
        assert uncovered_neuron is not None

        calc.update_coverage(np.array([[0.1, 0.1]]))
        calc.update_coverage(np.array([[0.8, 0.8]]))
        coverage = calc.get_coverage()
        assert coverage['covered_sections'] == 15
        assert coverage['sections_covered_percentage'] == 1

    def test_NeuronBoundaryCoverageCalculator(self):
        model = self._generate_simple_feedforward_model()
        calc = NeuronBoundaryCoverageCalculator(model)
        calc.update_neuron_bounds([np.array([[1, 1]])])
        calc.update_neuron_bounds([np.array([[0, 0]])])

        calc.update_coverage(np.array([[2, 0]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert coverage['total_corners'] == 10
        assert coverage['covered_corners'] == 1
        assert coverage['corners_covered_percentage'] == 1 / 10
        assert uncovered_neuron is not None

        calc.update_coverage(np.array([[2, 2]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert coverage['total_corners'] == 10
        assert coverage['covered_corners'] == 5
        assert uncovered_neuron is not None

        calc.update_coverage(np.array([[-2, -2]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert coverage['total_corners'] == 10
        assert coverage['covered_corners'] == 10
        assert coverage['covered_neurons'] == 5
        assert uncovered_neuron is not None

    def test_TopKNeuronCoverageCalculator(self):
        model = self._generate_simple_feedforward_model()
        calc = TopKNeuronCoverageCalculator(model, k=1)

        calc.update_coverage(np.array([[1, 0]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert uncovered_neuron is not None
        coverage['top_k_neurons'] = 3
        coverage['top_k_neurons_covered'] = 3 / 5

        calc.update_coverage(np.array([[0, 1]]))
        coverage = calc.get_coverage()
        uncovered_neuron = calc.get_random_uncovered_neuron()
        assert uncovered_neuron is not None
        coverage['top_k_neurons'] = 5
        coverage['top_k_neurons_covered'] = 1

    def test_TopKNeuronPatternsCalculator(self):
        model = self._generate_simple_feedforward_model()
        calc = TopKNeuronPatternsCalculator(model, k=1)

        calc.update_coverage(np.array([[1, 0]]))
        coverage = calc.get_coverage()
        assert coverage['total_patterns'] == 1

        calc.update_coverage(np.array([[0, 1]]))
        coverage = calc.get_coverage()
        assert coverage['total_patterns'] == 2

if __name__ == '__main__':
    for _ in range(10):
        unittest.main()
