import math

import numpy as np

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
