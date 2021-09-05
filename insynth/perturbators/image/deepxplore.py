import random
from collections import defaultdict
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow import keras

from insynth.input import ImageInput
from insynth.perturbation import WhiteboxImagePerturbator


class DeepXploreImagePerturbator(WhiteboxImagePerturbator):
    def __init__(self, model1, model2, model3, coverage_metric, snac_data):
        super().__init__(model1)
        self.model1=model1
        self.model2 = model2
        self.model3 = model3
        self.model_layer_dict1, self.model_layer_dict2, self.model_layer_dict3 = DeepXploreImagePerturbator.init_coverage_tables(
            model1, model2, model3)

        m1_dict, m2_dict, m3_dict = {}, {}, {}

        m1_dict["snac"], m2_dict["snac"], m3_dict["snac"] = DeepXploreImagePerturbator.init_coverage_tables(model1,
                                                                                                            model2,
                                                                                                            model3)
        m1_dict["snac_test"], m2_dict["snac_test"], m3_dict[
            "snac_test"] = DeepXploreImagePerturbator.init_coverage_tables(model1, model2, model3)
        m1_dict["nc"], m2_dict["nc"], m3_dict["nc"] = DeepXploreImagePerturbator.init_coverage_tables(model1, model2,
                                                                                                      model3)
        m1_dict["nc_test"], m2_dict["nc_test"], m3_dict["nc_test"] = DeepXploreImagePerturbator.init_coverage_tables(
            model1, model2, model3)
        self.m1_dict, self.m2_dict, self.m3_dict = m1_dict, m2_dict, m3_dict

        if coverage_metric == 'SNAC':
            for model, model_layer_dict in [(model1, self.model_layer_dict1), (model2, self.model_layer_dict2),
                                            (model3, self.model_layer_dict3)]:
                for img in snac_data:
                    gen_img = np.expand_dims(img, axis=0)
                    DeepXploreImagePerturbator.update_neuron_bounds(gen_img, model1, model_layer_dict)

    def apply(self, original_input: ImageInput):
        gen_img = np.expand_dims(original_input.image, axis=0)
        label1, label2, label3 = np.argmax(self.model1.predict(gen_img)[0]), np.argmax(self.model2.predict(gen_img)[0]), np.argmax(
            self.model3.predict(gen_img)[0])

        DeepXploreImagePerturbator.update_coverage(gen_img, self.model1, self.m1_dict, self.model_layer_dict1, True, 0)
        DeepXploreImagePerturbator.update_coverage(gen_img, self.model2, self.m2_dict, self.model_layer_dict2, True, 0)
        DeepXploreImagePerturbator.update_coverage(gen_img, self.model3, self.m3_dict, self.model_layer_dict3, True, 0)

        if not label1==label2==label3:
            return gen_img
        orig_label = label1



        # we run gradient ascent for 20 steps
        for iters in range(20):
            input_variable = tf.Variable(gen_img)

            before_softmax_int_model1 = keras.models.Model(inputs=self.model1.inputs, outputs=self.model1.get_layer('before_softmax').output)
            before_softmax_int_model2 = keras.models.Model(inputs=self.model2.inputs, outputs=self.model2.get_layer('before_softmax').output)
            before_softmax_int_model3 = keras.models.Model(inputs=self.model3.inputs, outputs=self.model3.get_layer('before_softmax').output)



            layer_name1, index1 = DeepXploreImagePerturbator.neuron_to_cover(self.m1_dict['nc'])
            layer_name2, index2 = DeepXploreImagePerturbator.neuron_to_cover(self.m2_dict['nc'])
            layer_name3, index3 = DeepXploreImagePerturbator.neuron_to_cover(self.m3_dict['nc'])

            int_model1 = keras.models.Model(inputs=self.model1.inputs, outputs=self.model1.get_layer(layer_name1).output)
            int_model2 = keras.models.Model(inputs=self.model2.inputs, outputs=self.model2.get_layer(layer_name2).output)
            int_model3 = keras.models.Model(inputs=self.model3.inputs, outputs=self.model3.get_layer(layer_name3).output)

            with tf.GradientTape() as tape:
                tape.watch(input_variable)

                loss1 = -1 * tf.math.reduce_mean(before_softmax_int_model1(input_variable)[..., orig_label])
                loss2 = tf.math.reduce_mean(before_softmax_int_model2(input_variable)[..., orig_label])
                loss3 = tf.math.reduce_mean(before_softmax_int_model3(input_variable)[..., orig_label])


                loss1_neuron = int_model1(input_variable)[0][np.unravel_index(index1,list(self.model1.get_layer(layer_name1).output.shape)[1:])]
                loss2_neuron = int_model2(input_variable)[0][np.unravel_index(index2,list(self.model2.get_layer(layer_name2).output.shape)[1:])]
                loss3_neuron = int_model3(input_variable)[0][np.unravel_index(index3,list(self.model3.get_layer(layer_name3).output.shape)[1:])]

                layer_output = (loss1 + loss2 + loss3) + 0.1 * (loss1_neuron + loss2_neuron + loss3_neuron)

                # for adversarial image generation
                final_loss = tf.math.reduce_mean(loss1_neuron)


                # we compute the gradient of the input picture wrt this loss
            grads = DeepXploreImagePerturbator.normalize(tape.gradient(final_loss, input_variable))

            grads_value = DeepXploreImagePerturbator.constraint_light(grads)  # constraint the gradients value

            gen_img += grads_value * 0.01
            predictions1 = np.argmax(self.model1.predict(gen_img)[0])
            predictions2 = np.argmax(self.model2.predict(gen_img)[0])
            predictions3= np.argmax(self.model3.predict(gen_img)[0])



            if not predictions1 == predictions2 == predictions3:
                DeepXploreImagePerturbator.update_coverage(gen_img, self.model1, self.m1_dict, self.model_layer_dict1, 0)
                DeepXploreImagePerturbator.update_coverage(gen_img, self.model2, self.m2_dict, self.model_layer_dict2, 0)
                DeepXploreImagePerturbator.update_coverage(gen_img, self.model3, self.m3_dict, self.model_layer_dict3, 0)
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
    def init_coverage_tables(model1, model2, model3):
        model_layer_dict1 = defaultdict(bool)
        model_layer_dict2 = defaultdict(bool)
        model_layer_dict3 = defaultdict(bool)
        DeepXploreImagePerturbator.init_dict(model1, model_layer_dict1)
        DeepXploreImagePerturbator.init_dict(model2, model_layer_dict2)
        DeepXploreImagePerturbator.init_dict(model3, model_layer_dict3)
        return model_layer_dict1, model_layer_dict2, model_layer_dict3

    @staticmethod
    def init_dict(model, model_layer_dict):
        for layer in model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            for index in range(DeepXploreImagePerturbator.num_neurons(layer.output_shape)):  # product of dims
                model_layer_dict[(layer.name, index)] = False

    @staticmethod
    def neuron_to_cover(model_layer_dict):
        not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
        if not_covered:
            layer_name, index = random.choice(not_covered)
        else:
            layer_name, index = random.choice(model_layer_dict.keys())
        return layer_name, index

    @staticmethod
    def neuron_covered(model_layer_dict):
        covered_neurons = len([v for v in model_layer_dict.values() if v])
        total_neurons = len(model_layer_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    @staticmethod
    def update_coverage(input_data, model, model_layer_dict, model_layer_hl_dict, test_only=False, threshold=0):
        snac_dict, nc_dict = {}, {}
        if test_only:
            snac_dict = model_layer_dict["snac_test"]
            nc_dict = model_layer_dict["nc_test"]
        else:
            snac_dict = model_layer_dict["snac"]
            nc_dict = model_layer_dict["nc"]

        layer_names = [layer.name for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name]

        intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                      outputs=[model.get_layer(layer_name).output for layer_name in
                                                               layer_names])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            layer = intermediate_layer_output[0]
            for neuron in range(
                    DeepXploreImagePerturbator.num_neurons(layer.shape)):  # index through every single (indiv) neuron
                _, high = model_layer_hl_dict[(layer_names[i], neuron)]

                # evaluate snac criteria
                if layer[np.unravel_index(neuron, layer.shape)] > high and not snac_dict[(layer_names[i], neuron)]:
                    snac_dict[(layer_names[i], neuron)] = True

                # evaluate nc criteria
                if layer[np.unravel_index(neuron, layer.shape)] > threshold and not nc_dict[(layer_names[i], neuron)]:
                    nc_dict[(layer_names[i], neuron)] = True

    @staticmethod
    def num_neurons(shape):
        return reduce(lambda x, y: x * y, filter(lambda x: x != None, shape))

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
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
        scaled = DeepXploreImagePerturbator.scale(intermediate_layer_output)
        if np.mean(scaled[..., index]) > threshold:
            return True
        return False

    @staticmethod
    def diverged(predictions1, predictions2, predictions3, target):
        if not predictions1 == predictions2 == predictions3:
            return True
        return False

    @staticmethod
    def update_neuron_bounds(input_data, model, model_layer_dict):
        layer_names = [layer.name for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name]

        intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                      outputs=[model.get_layer(layer_name).output for layer_name in
                                                               layer_names])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            layer = intermediate_layer_output[0]
            for neuron in range(
                    DeepXploreImagePerturbator.num_neurons(layer.shape)):  # index through every single (indiv) neuron
                v = layer[np.unravel_index(neuron, layer.shape)]

                if not model_layer_dict[(layer_names[i], neuron)]:  # get rid of mean
                    model_layer_dict[(layer_names[i], neuron)] = (v, v)
                else:
                    (lower, upper) = model_layer_dict[(layer_names[i], neuron)]
                    if v > upper:
                        model_layer_dict[(layer_names[i], neuron)] = (lower, v)
                    elif v < lower:
                        model_layer_dict[(layer_names[i], neuron)] = (v, upper)
