import functools
import math
from audiomentations.augmentations.transforms import AddBackgroundNoise,  ApplyImpulseResponse
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import get_file_paths

import numpy as np
from audiomentations import AddGaussianNoise, Mp3Compression, PitchShift, ClippingDistortion, Gain, AddShortNoises
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


class AudioEchoPerturbator(BlackboxAudioPerturbator):
    def apply(self, original_input):
        signal, sample_rate = original_input
        output_audio = np.zeros(len(signal))
        output_delay = 1.0 * sample_rate

        for count, e in enumerate(signal):
            output_audio[count] = e + signal[count - int(output_delay)]

        return output_audio


class AudioShortNoisePerturbator(BlackboxAudioPerturbator):
    def __init__(self, p=1.0, types=[]) -> None:
        super().__init__()
        self.p = p
        self.types = types
        self.sound_file_paths = []
        for type in types:
            self.sound_file_paths.extend(get_file_paths(
                f'data/audio/background_noise/{type}'))
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]
        
    def apply(self, original_input):
        signal, sample_rate = original_input
        op = AddShortNoises(
            sounds_path='data/audio/background_noise/esc-50/', p=1.0)
        op.sound_file_paths=self.sound_file_paths
        return op(signal, sample_rate=sample_rate)


class AudioBackgroundNoisePerturbator(BlackboxAudioPerturbator):
    def __init__(self, p=1.0, types=[]) -> None:
        super().__init__()
        self.p = p
        self.types = types
        self.sound_file_paths = []
        for type in types:
            self.sound_file_paths.extend(get_file_paths(
                f'data/audio/background_noise/{type}'))
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]

    def apply(self, original_input):
        signal, sample_rate = original_input

        op = AddBackgroundNoise(
            sounds_path='data/audio/background_noise/esc-50/', p=self.p)
        op.sound_file_paths=self.sound_file_paths
        return op(signal, sample_rate=sample_rate)


class AudioImpulseResponsePerturbator(BlackboxAudioPerturbator):

    def __init__(self, p=1.0, types=[]) -> None:
        super().__init__()
        self.p = p
        self.types = types
        self.ir_files = []
        for type in types:
            self.ir_files.extend(get_file_paths(
                f'data/audio/pulse_response/{type}'))
        self.ir_files = [str(p) for p in self.ir_files]

    def apply(self, original_input):
        signal, sample_rate = original_input
        op = ApplyImpulseResponse(ir_path='data/audio/pulse_response/', p=self.p)
        op.ir_files = self.ir_files
        return op(signal, sample_rate)
