import random
from collections import defaultdict
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow import keras

from insynth.input import ImageInput
from insynth.metrics.coverage.coverage_calculator import AbstractCoverageCalculator
from insynth.metrics.coverage.neuron import NeuronCoverageCalculator, StrongNeuronActivationCoverageCalculator
from insynth.perturbation import WhiteboxImagePerturbator

COVERAGE_CRITERIA_TO_CALCULATOR_CLASS = {
    'NC': NeuronCoverageCalculator,
    'SNAC': StrongNeuronActivationCoverageCalculator
}


class DeepXploreImagePerturbator(WhiteboxImagePerturbator):
    def __init__(self, model1, model2, model3, coverage_criteria, snac_data):
        super().__init__(model1)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.coverage_criteria = coverage_criteria

        calculator_class = COVERAGE_CRITERIA_TO_CALCULATOR_CLASS[coverage_criteria]

        self.model1_coverage_calculator = calculator_class(self.model1)
        self.model2_coverage_calculator = calculator_class(self.model2)
        self.model3_coverage_calculator = calculator_class(self.model3)

        if coverage_criteria == 'SNAC':
            self.model1_coverage_calculator.update_neuron_bounds(snac_data)
            self.model2_coverage_calculator.update_neuron_bounds(snac_data)
            self.model3_coverage_calculator.update_neuron_bounds(snac_data)

    def apply(self, original_input, force_mutation=False):
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
                                                           outputs=self.model1.get_layer('before_softmax').output)
            before_softmax_int_model2 = keras.models.Model(inputs=self.model2.inputs,
                                                           outputs=self.model2.get_layer('before_softmax').output)
            before_softmax_int_model3 = keras.models.Model(inputs=self.model3.inputs,
                                                           outputs=self.model3.get_layer('before_softmax').output)

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
            grads = DeepXploreImagePerturbator.normalize(tape.gradient(final_loss, input_variable))

            grads_value = DeepXploreImagePerturbator.constraint_light(grads)  # constraint the gradients value

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
    def deprocess_image(x):
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)

    @staticmethod
    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (tf.math.sqrt(tf.math.reduce_mean(tf.math.square(x))) + 1e-5)

    @staticmethod
    def constraint_occl(gradients, start_point, rect_shape):
        new_grads = np.zeros_like(gradients)
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                         start_point[1]:start_point[1] + rect_shape[1]]
        return new_grads

    @staticmethod
    def constraint_light(gradients):
        new_grads = np.ones_like(gradients)
        grad_mean = np.mean(gradients)
        return grad_mean * new_grads

    @staticmethod
    def constraint_black(gradients, rect_shape=(6, 6)):
        start_point = (
            random.randint(0, gradients.shape[1] - rect_shape[0]),
            random.randint(0, gradients.shape[2] - rect_shape[1]))
        new_grads = np.zeros_like(gradients)
        patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                start_point[1]:start_point[1] + rect_shape[1]]
        if np.mean(patch) < 0:
            new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
            start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
        return new_grads

    @staticmethod
    def full_coverage(model_layer_dict):
        if False in model_layer_dict.values():
            return False
        return True

    @staticmethod
    def scale(intermediate_layer_output, rmax=1, rmin=0):
        X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
                intermediate_layer_output.max() - intermediate_layer_output.min())
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    @staticmethod
    def fired(model, layer_name, index, input_data, threshold=0):
        intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
        scaled = DeepXploreImagePerturbator.scale(intermediate_layer_output)
        if np.mean(scaled[..., index]) > threshold:
            return True
        return False
