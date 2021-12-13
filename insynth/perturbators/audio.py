import math
import random

import numpy as np
from audiomentations import Mp3Compression, PitchShift, ClippingDistortion, Gain, AddShortNoises
from audiomentations.augmentations.transforms import AddBackgroundNoise, ApplyImpulseResponse
from audiomentations.core.utils import get_file_paths
from scipy.stats import norm

from insynth.data import utils
from insynth.perturbation import BlackboxAudioPerturbator, WhiteboxAudioPerturbator


class AudioBackgroundWhiteNoisePerturbator(BlackboxAudioPerturbator):
    def __init__(self, p=0.5, noise_prob=norm, noise_prob_args={'loc': 0.2, 'scale': 0.2}) -> None:
        super().__init__(p=p)
        self.noise_prob = noise_prob
        self.noise_prob_args = noise_prob_args

    def _internal_apply(self, original_input):
        signal, sample_rate = original_input

        noise_level = self.noise_prob.rvs(**self.noise_prob_args)
        noise_level = max(min(1.0, noise_level), 0.0)

        RMS = math.sqrt(np.mean(signal ** 2))
        noise = np.random.normal(0, RMS * noise_level, signal.shape[0])
        signal_noise = signal + noise
        return signal_noise



class AudioCompressionPerturbator(BlackboxAudioPerturbator):
    def __init__(self, p=0.5, compression_prob=norm, compression_prob_args={'loc': 80, 'scale': 40}) -> None:
        super().__init__(p=p)
        self.compression_prob = compression_prob
        self.compression_prob_args = compression_prob_args

    def _internal_apply(self, original_input):
        signal, sample_rate = original_input

        compression_rate = self.compression_prob.rvs(
            **self.compression_prob_args)
        compression_rate = min(
            Mp3Compression.SUPPORTED_BITRATES, key=lambda x: abs(x - compression_rate))

        op = Mp3Compression(p=1.0, min_bitrate=compression_rate,
                            max_bitrate=compression_rate, backend='lameenc')
        return op(signal, sample_rate)


class AudioPitchPerturbator(BlackboxAudioPerturbator):
    def __init__(self, p=0.5, pitch_prob=norm, pitch_prob_args={'loc': 0, 'scale': 8}) -> None:
        super().__init__(p=p)
        self.pitch_prob = pitch_prob
        self.pitch_prob_args = pitch_prob_args

    def _internal_apply(self, original_input):
        signal, sample_rate = original_input

        pitch_shift = self.pitch_prob.rvs(**self.pitch_prob_args)
        pitch_shift = int(max(min(12, pitch_shift), -12))

        op = PitchShift(p=1.0, min_semitones=pitch_shift,
                        max_semitones=pitch_shift)
        return op(signal, sample_rate)


class AudioClippingPerturbator(BlackboxAudioPerturbator):
    def __init__(self, p=0.5, clipping_prob=norm, clipping_prob_args={'loc': 20, 'scale': 30}) -> None:
        super().__init__(p=p)
        self.clipping_prob = clipping_prob
        self.clipping_prob_args = clipping_prob_args

    def _internal_apply(self, original_input):
        signal, sample_rate = original_input

        clipping_percentile = self.clipping_prob.rvs(**self.clipping_prob_args)
        clipping_percentile = int(max(min(80, clipping_percentile), 0))

        op = ClippingDistortion(
            p=1.0, min_percentile_threshold=clipping_percentile, max_percentile_threshold=clipping_percentile)
        return op(signal, sample_rate)


class AudioVolumePerturbator(BlackboxAudioPerturbator):
    def __init__(self, p=0.5, volume_prob=norm, volume_prob_args={'loc': 0, 'scale': 10}) -> None:
        super().__init__(p=p)
        self.volume_prob = volume_prob
        self.volume_prob_args = volume_prob_args

    def _internal_apply(self, original_input):
        signal, sample_rate = original_input

        volume_shift = self.volume_prob.rvs(**self.volume_prob_args)
        volume_shift = max(min(20, volume_shift), -20)

        op = Gain(p=1.0, min_gain_in_db=volume_shift,
                  max_gain_in_db=volume_shift)
        return op(signal, sample_rate)


class AudioEchoPerturbator(BlackboxAudioPerturbator):
    def __init__(self, p=0.5, echo_prob=norm, echo_prob_args={'loc': 0.3, 'scale': 0.1}) -> None:
        super().__init__(p=p)
        self.echo_prob = echo_prob
        self.echo_prob_args = echo_prob_args

    def _internal_apply(self, original_input):
        signal, sample_rate = original_input

        echo_delay = self.echo_prob.rvs(**self.echo_prob_args)
        echo_delay = max(min(1.0, echo_delay), 0.0)

        output_audio = np.zeros(len(signal))
        output_delay = echo_delay * sample_rate

        for count, e in enumerate(signal):
            output_audio[count] = e + signal[count - int(output_delay)]

        return output_audio


class AudioShortNoisePerturbator(BlackboxAudioPerturbator):
    def __init__(self, p=0.5, noise_types=['']) -> None:
        super().__init__()
        utils.download_and_unzip(
            'https://insynth-data.s3.eu-central-1.amazonaws.com/background_noise.zip',
            'data/audio/background_noise/')
        self.p = p
        self.noise_types = noise_types
        self.sound_file_paths = []
        for type in noise_types:
            self.sound_file_paths.extend(get_file_paths(
                f'data/audio/background_noise/{type}'))
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]

    def _internal_apply(self, original_input):
        signal, sample_rate = original_input

        op = AddShortNoises(
            sounds_path='data/audio/background_noise/', p=1.0)
        op.sound_file_paths = self.sound_file_paths  # overwrite files to sample from
        return op(signal, sample_rate=sample_rate)


class AudioBackgroundNoisePerturbator(BlackboxAudioPerturbator):
    def __init__(self, p=0.5, noise_types=['']) -> None:
        super().__init__()
        utils.download_and_unzip(
            'https://insynth-data.s3.eu-central-1.amazonaws.com/background_noise.zip',
            'data/audio/background_noise/')
        self.p = p
        self.noise_types = noise_types
        self.sound_file_paths = []
        for type in noise_types:
            self.sound_file_paths.extend(get_file_paths(
                f'data/audio/background_noise/{type}'))
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]

    def _internal_apply(self, original_input):
        signal, sample_rate = original_input

        op = AddBackgroundNoise(
            sounds_path='data/audio/background_noise/', p=self.p)
        op.sound_file_paths = self.sound_file_paths
        return op(signal, sample_rate=sample_rate)


class AudioImpulseResponsePerturbator(BlackboxAudioPerturbator):

    def __init__(self, p=0.5, impulse_types=['']) -> None:
        utils.download_and_unzip(
            'https://insynth-data.s3.eu-central-1.amazonaws.com/impulse_response.zip',
            'data/audio/pulse_response/')
        super().__init__()
        self.p = p
        self.impulse_types = impulse_types
        self.ir_files = []
        for type in impulse_types:
            self.ir_files.extend(get_file_paths(
                f'data/audio/pulse_response/{type}'))
        self.ir_files = [str(p) for p in self.ir_files]

    def _internal_apply(self, original_input):
        signal, sample_rate = original_input

        op = ApplyImpulseResponse(
            ir_path='data/audio/pulse_response/', p=self.p)
        op.ir_files = self.ir_files
        return op(signal, sample_rate)
