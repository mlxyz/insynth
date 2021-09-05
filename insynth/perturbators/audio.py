import math

import numpy as np
from scipy.io.wavfile import write

from insynth.input import AudioInput
from insynth.perturbation import BlackboxAudioPerturbator


class AudioBackgroundWhiteNoisePerturbator(BlackboxAudioPerturbator):
    def apply(self, original_input: AudioInput, noise_level=0.1):
        RMS = math.sqrt(np.mean(original_input.signal ** 2))
        noise = np.random.normal(0, RMS * noise_level, original_input.signal.shape[0])
        signal_noise = original_input.signal + noise

        write('test.wav', original_input.sr, signal_noise)