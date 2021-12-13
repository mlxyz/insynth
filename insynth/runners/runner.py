import re
import string
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from PIL import Image
from keras import layers
from sklearn.metrics import classification_report

from insynth.metrics.coverage.neuron import StrongNeuronActivationCoverageCalculator, \
    KMultiSectionNeuronCoverageCalculator, NeuronCoverageCalculator, NeuronBoundaryCoverageCalculator, \
    TopKNeuronCoverageCalculator, TopKNeuronPatternsCalculator
from insynth.perturbators.audio import AudioBackgroundWhiteNoisePerturbator, AudioCompressionPerturbator, \
    AudioPitchPerturbator, AudioClippingPerturbator, AudioVolumePerturbator, AudioEchoPerturbator, \
    AudioShortNoisePerturbator, AudioBackgroundNoisePerturbator, AudioImpulseResponsePerturbator
from insynth.perturbators.image import ImageNoisePerturbator, ImageBrightnessPerturbator, ImageContrastPerturbator, \
    ImageSharpnessPerturbator, ImageFlipPerturbator, ImageOcclusionPerturbator, ImageCompressionPerturbator, \
    ImagePixelizePerturbator

import tensorflow as tf

from insynth.perturbators.text import TextTypoPerturbator, TextCasePerturbator, TextWordRemovalPerturbator, \
    TextStopWordRemovalPerturbator, TextWordSwitchPerturbator, TextCharacterSwitchPerturbator, \
    TextPunctuationErrorPerturbator


class AbstractRunner(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError


class BasicRunner(AbstractRunner):

    def __init__(self, perturbators, coverage_calculators, dataset_x, dataset_y, model):
        self.perturbators = perturbators
        self.coverage_calculators = coverage_calculators
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.model = model

    def run(self, save_mutated_samples=False, output_path=None):
        results = {}
        original_dataset = self._pre_prediction(self.dataset_x)
        y_pred = np.argmax(self.model.predict(original_dataset, verbose=1), axis=1)

        self.put_results_into_dict(results, 'original', self.dataset_y, y_pred)

        for perturbator_index, perturbator in enumerate(self.perturbators):
            perturbator_name = type(perturbator).__name__
            mutated_samples = self._apply_perturbator(self.dataset_x, perturbator)
            if save_mutated_samples:
                for sample_index, mutated_sample in enumerate(mutated_samples):
                    self._save(mutated_sample, f'{output_path}/{perturbator_name}_{sample_index}')

            perturbated_dataset = self._pre_prediction(mutated_samples)

            predictions = np.argmax(self.model.predict(perturbated_dataset, verbose=1), axis=1)

            self.put_results_into_dict(results, perturbator_name, self.dataset_y, predictions)
        return pd.DataFrame.from_dict(results, orient='index')

    def put_results_into_dict(self, dct, name, y_true, y_pred):
        results = classification_report(y_true,
                                        y_pred,
                                        output_dict=True)
        dct[name] = {
            'acc': results['accuracy'],
            'macro_f1': results['macro avg']['f1-score'],
            'macro_rec': results['macro avg']['recall'],
            'macro_prec': results['macro avg']['precision'],
            'micro_f1': results['weighted avg']['f1-score'],
            'micro_rec': results['weighted avg']['recall'],
            'micro_prec': results['weighted avg']['precision']}

    @abstractmethod
    def _apply_perturbator(self, samples, perturbator):
        raise NotImplementedError()

    @abstractmethod
    def _save(self, sample, output_path):
        raise NotImplementedError()

    def _pre_prediction(self, samples):
        return np.array([np.array(sample) for sample in samples])


class BasicImageRunner(BasicRunner):
    def _apply_perturbator(self, samples, perturbator):
        return [perturbator.apply(Image.fromarray(sample)) for sample in samples]

    def _save(self, sample, output_path):
        sample.save(output_path + '.jpg', 'JPEG')


class BasicTextRunner(BasicRunner):
    def _apply_perturbator(self, samples, perturbator):
        return list(map(perturbator.apply, samples))

    def _save(self, sample, output_path):
        with open(output_path + '.txt', 'w') as txt_out:
            txt_out.write(sample)

    def custom_standardization(self,input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')

    def _pre_prediction(self, samples):
        vectorize_layer = layers.TextVectorization(
            standardize=self.custom_standardization,
            max_tokens=10000,
            output_mode='int',
            output_sequence_length=250)
        vectorize_layer.adapt(samples)
        return vectorize_layer(samples).numpy()


class BasicAudioRunner(BasicRunner):
    def __init__(self, perturbators, coverage_calculators, dataset_x, dataset_y, model):
        super().__init__(perturbators, coverage_calculators, dataset_x, dataset_y, model)

    def _apply_perturbator(self, samples, perturbator):
        return [(perturbator.apply(sample), sample[1]) for sample in samples]

    def _save(self, sample, output_path):
        from scipy.io.wavfile import write
        signal, sample_rate = sample
        write(output_path + '.wav', sample_rate, signal)

    def _pre_prediction(self, samples):
        import tensorflow as tf

        out = []
        for sample in samples:
            signal = np.expand_dims(sample[0], axis=1)
            sample_rate = sample[1]
            audio = signal
            fft = tf.signal.fft(
                tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
            )
            # fft = tf.expand_dims(fft, axis=0)

            # Return the absolute value of the first half of the FFT
            # which represents the positive frequencies
            out.append(tf.math.abs(fft[: (sample_rate // 2), :]).numpy())
        return np.array(out)


class ComprehensiveImageRunner(BasicImageRunner):
    def __init__(self, dataset_x, dataset_y, model):
        super().__init__(self._get_all_perturbators(), self._get_all_coverage_calculators(model), dataset_x, dataset_y,
                         model)

    def _get_all_perturbators(self):
        return [ImageNoisePerturbator(p=1.0),
                ImageBrightnessPerturbator(p=1.0),
                ImageContrastPerturbator(p=1.0),
                ImageSharpnessPerturbator(p=1.0),
                ImageFlipPerturbator(p=1.0),
                ImageOcclusionPerturbator(p=1.0),
                ImageCompressionPerturbator(p=1.0),
                ImagePixelizePerturbator(p=1.0)
                ]

    def _get_all_coverage_calculators(self, model):
        return [
            NeuronCoverageCalculator(model),
            StrongNeuronActivationCoverageCalculator(model),
            KMultiSectionNeuronCoverageCalculator(model),
            NeuronBoundaryCoverageCalculator(model),
            TopKNeuronCoverageCalculator(model),
            TopKNeuronPatternsCalculator(model),
        ]


class ComprehensiveAudioRunner(BasicAudioRunner):
    def __init__(self, dataset_x, dataset_y, model):
        super().__init__(self._get_all_perturbators(), self._get_all_coverage_calculators(model), dataset_x, dataset_y,
                         model)

    def _get_all_perturbators(self):
        return [AudioBackgroundWhiteNoisePerturbator(p=1.0),
                # AudioCompressionPerturbator(p=1.0),
                AudioPitchPerturbator(p=1.0),
                AudioClippingPerturbator(p=1.0),
                AudioVolumePerturbator(p=1.0),
                AudioEchoPerturbator(p=1.0),
                AudioShortNoisePerturbator(p=1.0),
                AudioBackgroundNoisePerturbator(p=1.0),
                AudioImpulseResponsePerturbator(p=1.0)
                ]

    def _get_all_coverage_calculators(self, model):
        return [
            NeuronCoverageCalculator(model),
            StrongNeuronActivationCoverageCalculator(model),
            KMultiSectionNeuronCoverageCalculator(model),
            NeuronBoundaryCoverageCalculator(model),
            TopKNeuronCoverageCalculator(model),
            TopKNeuronPatternsCalculator(model),
        ]


class ComprehensiveTextRunner(BasicTextRunner):
    def __init__(self, dataset_x, dataset_y, model):
        super().__init__(self._get_all_perturbators(), self._get_all_coverage_calculators(model), dataset_x, dataset_y,
                         model)

    def _get_all_perturbators(self):
        return [TextTypoPerturbator(p=1.0),
                TextCasePerturbator(p=1.0),
                TextWordRemovalPerturbator(p=1.0),
                TextStopWordRemovalPerturbator(p=1.0),
                TextWordSwitchPerturbator(p=1.0),
                TextCharacterSwitchPerturbator(p=1.0),
                TextPunctuationErrorPerturbator(p=1.0),
                ]

    def _get_all_coverage_calculators(self, model):
        return [
            NeuronCoverageCalculator(model),
            StrongNeuronActivationCoverageCalculator(model),
            KMultiSectionNeuronCoverageCalculator(model),
            NeuronBoundaryCoverageCalculator(model),
            TopKNeuronCoverageCalculator(model),
            TopKNeuronPatternsCalculator(model)]
