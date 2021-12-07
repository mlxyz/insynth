from abc import ABC, abstractmethod
import random

import numpy as np

from insynth.metrics.coverage.neuron import NeuronCoverageCalculator, StrongNeuronActivationCoverageCalculator, \
    NeuronBoundaryCoverageCalculator, KMultiSectionNeuronCoverageCalculator, TopKNeuronCoverageCalculator, \
    TopKNeuronPatternsCalculator


class AbstractBlackboxPerturbator(ABC):
    def __init__(self, p=0.5):
        self.p = p

    @abstractmethod
    def apply(self, original_input):
        pass


class BlackboxImagePerturbator(AbstractBlackboxPerturbator):
    def __init__(self, p=0.5):
        super().__init__(p)

    @abstractmethod
    def apply(self, original_input):
        pass


class BlackboxAudioPerturbator(AbstractBlackboxPerturbator):
    def __init__(self, p=0.5):
        super().__init__(p)

    @abstractmethod
    def apply(self, original_input):
        pass


class BlackboxTextPerturbator(AbstractBlackboxPerturbator):
    def __init__(self, p=0.5):
        super().__init__(p)

    def apply(self, original_input):
        if random.random() > self.p:
            return original_input
        return self._internal_apply(original_input)

    @abstractmethod
    def _internal_apply(self, original_input):
        pass


class AbstractWhiteboxPerturbator(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def apply(self, original_input):
        pass


class WhiteboxImagePerturbator(AbstractWhiteboxPerturbator):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def apply(self, original_input):
        pass


class WhiteboxAudioPerturbator(AbstractWhiteboxPerturbator):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def apply(self, original_input):
        pass


class WhiteboxTextPerturbator(AbstractWhiteboxPerturbator):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def apply(self, original_input):
        pass


COVERAGE_CRITERIA_TO_CALCULATOR_CLASS = {
    'NC': NeuronCoverageCalculator,
    'SNAC': StrongNeuronActivationCoverageCalculator,
    'NBC': NeuronBoundaryCoverageCalculator,
    'KMSNC': KMultiSectionNeuronCoverageCalculator,
    'TKNC': TopKNeuronCoverageCalculator,
    'TKPC': TopKNeuronPatternsCalculator
}


class GenericDeepXplorePerturbator(AbstractWhiteboxPerturbator):

    def __init__(self, model1, model2, model3, coverage_criteria, snac_data=None):
        super().__init__(model1)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.coverage_criteria = coverage_criteria

        calculator_class = COVERAGE_CRITERIA_TO_CALCULATOR_CLASS[coverage_criteria]

        self.model1_coverage_calculator = calculator_class(self.model1)
        self.model2_coverage_calculator = calculator_class(self.model2)
        self.model3_coverage_calculator = calculator_class(self.model3)

        if coverage_criteria != 'NC' and coverage_criteria != 'TKNC' and coverage_criteria != 'TKPC':
            for image in snac_data:
                self.model1_coverage_calculator.update_neuron_bounds(image)
                self.model2_coverage_calculator.update_neuron_bounds(image)
                self.model3_coverage_calculator.update_neuron_bounds(image)

    def apply(self, original_input, force_mutation=False):
        from tensorflow import keras
        import tensorflow as tf
        gen_img = original_input
        label1, label2, label3 = np.argmax(self.model1.predict(gen_img)[0]), np.argmax(
            self.model2.predict(gen_img)[0]), np.argmax(
            self.model3.predict(gen_img)[0])

        self.model1_coverage_calculator.update_coverage(gen_img)
        self.model2_coverage_calculator.update_coverage(gen_img)
        self.model3_coverage_calculator.update_coverage(gen_img)

        if not force_mutation and not label1 == label2 == label3:
            return gen_img
        orig_label = label1

        # we run gradient ascent for 20 steps
        for _ in range(20):
            input_variable = tf.Variable(gen_img)

            before_softmax_int_model1 = keras.models.Model(inputs=self.model1.inputs,
                                                           outputs=self.model1.layers[-1].output)
            before_softmax_int_model2 = keras.models.Model(inputs=self.model2.inputs,
                                                           outputs=self.model2.layers[-1].output)
            before_softmax_int_model3 = keras.models.Model(inputs=self.model3.inputs,
                                                           outputs=self.model3.layers[-1].output)

            layer_name1, index1 = self.model1_coverage_calculator.get_random_uncovered_neuron()
            layer_name2, index2 = self.model2_coverage_calculator.get_random_uncovered_neuron()
            layer_name3, index3 = self.model3_coverage_calculator.get_random_uncovered_neuron()

            int_model1 = keras.models.Model(inputs=self.model1.inputs,
                                            outputs=self.model1.get_layer(layer_name1).output)
            int_model2 = keras.models.Model(inputs=self.model2.inputs,
                                            outputs=self.model2.get_layer(layer_name2).output)
            int_model3 = keras.models.Model(inputs=self.model3.inputs,
                                            outputs=self.model3.get_layer(layer_name3).output)

            with tf.GradientTape() as tape:
                tape.watch(input_variable)

                loss1 = -1 * tf.math.reduce_mean(before_softmax_int_model1(input_variable)[..., orig_label])
                loss2 = tf.math.reduce_mean(before_softmax_int_model2(input_variable)[..., orig_label])
                loss3 = tf.math.reduce_mean(before_softmax_int_model3(input_variable)[..., orig_label])

                loss1_neuron = int_model1(input_variable)[0][
                    np.unravel_index(index1, list(self.model1.get_layer(layer_name1).output.shape)[1:])]
                loss2_neuron = int_model2(input_variable)[0][
                    np.unravel_index(index2, list(self.model2.get_layer(layer_name2).output.shape)[1:])]
                loss3_neuron = int_model3(input_variable)[0][
                    np.unravel_index(index3, list(self.model3.get_layer(layer_name3).output.shape)[1:])]

                layer_output = (loss1 + loss2 + loss3) + 0.1 * (loss1_neuron + loss2_neuron + loss3_neuron)

                # for adversarial image generation
                final_loss = tf.math.reduce_mean(layer_output)

                # we compute the gradient of the input picture wrt this loss
            grads = GenericDeepXplorePerturbator.normalize(tape.gradient(final_loss, input_variable))

            grads_value = self.apply_gradient_constraint(grads)  # constraint the gradients value

            gen_img += grads_value * 0.01
            predictions1 = np.argmax(self.model1.predict(gen_img)[0])
            predictions2 = np.argmax(self.model2.predict(gen_img)[0])
            predictions3 = np.argmax(self.model3.predict(gen_img)[0])

            self.model1_coverage_calculator.update_coverage(gen_img)
            self.model2_coverage_calculator.update_coverage(gen_img)
            self.model3_coverage_calculator.update_coverage(gen_img)

            if not force_mutation and not predictions1 == predictions2 == predictions3:
                return gen_img
        return gen_img

    @staticmethod
    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (tf.math.sqrt(tf.math.reduce_mean(tf.math.square(x))) + 1e-5)

    @abstractmethod
    def apply_gradient_constraint(self, grads):
        pass
