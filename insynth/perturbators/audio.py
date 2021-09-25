import math

import numpy as np
from audiomentations import AddGaussianNoise, Mp3Compression, PitchShift, ClippingDistortion, Clip, Gain
from insynth.perturbation import BlackboxAudioPerturbator, GenericDeepXplorePerturbator, WhiteboxAudioPerturbator


class AudioBackgroundWhiteNoisePerturbator(BlackboxAudioPerturbator):
    def apply(self, original_input, noise_level=0.1):
        signal = original_input
        RMS = math.sqrt(np.mean(signal ** 2))
        noise = np.random.normal(0, RMS * noise_level, signal.shape[0])
        signal_noise = signal + noise
        return signal_noise


class DeepXploreAudioPerturbator(GenericDeepXplorePerturbator, WhiteboxAudioPerturbator):
    def apply_gradient_constraint(self, grads):
        return grads


class AudioGaussianBackgroundNoisePerturbator(BlackboxAudioPerturbator):
    def apply(self, original_input):
        signal, sample_rate = original_input
        op = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)
        return op(signal, sample_rate=sample_rate)


class AudioCompressionPerturbator(BlackboxAudioPerturbator):
    def apply(self, original_input):
        signal, sample_rate = original_input
        op = Mp3Compression(p=1.0, min_bitrate=8, max_bitrate=8)
        return op(signal, sample_rate)


class AudioPitchPerturbator(BlackboxAudioPerturbator):
    def apply(self, original_input):
        signal, sample_rate = original_input
        op = PitchShift(p=1.0, min_semitones=-4, max_semitones=-4)
        return op(signal, sample_rate)


class AudioClippingPerturbator(BlackboxAudioPerturbator):
    def apply(self, original_input):
        signal, sample_rate = original_input
        op = ClippingDistortion(p=1.0)
        return op(signal, sample_rate)


class AudioVolumePerturbator(BlackboxAudioPerturbator):
    def apply(self, original_input):
        signal, sample_rate = original_input
        op = Gain(p=1.0)
        return op(signal, sample_rate)


class EchoPerturbator(BlackboxAudioPerturbator):
    def apply(self, original_input):
        signal, sample_rate = original_input
        output_audio = np.zeros(len(signal))
        output_delay = 1.0 * sample_rate

        for count, e in enumerate(signal):
            output_audio[count] = e + signal[count - int(output_delay)]

        return output_audio
