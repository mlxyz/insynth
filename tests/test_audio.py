import unittest

import numpy as np

from insynth.perturbators.audio import AudioBackgroundWhiteNoisePerturbator, AudioPitchPerturbator, \
    AudioClippingPerturbator, AudioVolumePerturbator, AudioEchoPerturbator, AudioShortNoisePerturbator, \
    AudioImpulseResponsePerturbator, AudioBackgroundNoisePerturbator


class TestAudio(unittest.TestCase):

    def _generate_random_audio(self):
        data = np.random.uniform(-1, 1, 44100)
        return data

    def test_AudioBackgroundWhiteNoisePerturbator_with_noise(self):
        input_signal = self._generate_random_audio()

        perturbator = AudioBackgroundWhiteNoisePerturbator(p=1.0,
                                                           noise_prob=type('', (object,), {'rvs': lambda _: 1.0})(),
                                                           noise_prob_args={})

        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        # assert arrays are not equal
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_signal, output_signal)

    def test_AudioBackgroundWhiteNoisePerturbator_without_noise(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioBackgroundWhiteNoisePerturbator(p=1.0,
                                                           noise_prob=type('', (object,), {'rvs': lambda _: 0.0})(),
                                                           noise_prob_args={})
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_array_equal(input_signal, output_signal)

    def test_AudioPitchPerturbator_with_pitch_change(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioPitchPerturbator(p=1.0,
                                            pitch_prob=type('', (object,), {'rvs': lambda _: 12})(),
                                            pitch_prob_args={})
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_signal, output_signal)

    def test_AudioPitchPerturbator_without_pitch_change(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioPitchPerturbator(p=1.0,
                                            pitch_prob=type('', (object,), {'rvs': lambda _: 0})(),
                                            pitch_prob_args={})
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_array_almost_equal(input_signal, output_signal, 1)

    def test_AudioClippingPerturbator_with_clipping(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioClippingPerturbator(p=1.0,
                                               clipping_prob=type('', (object,), {'rvs': lambda _: 50})(),
                                               clipping_prob_args={})
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_signal, output_signal)

    def test_AudioClippingPerturbator_without_clipping(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioClippingPerturbator(p=1.0,
                                               clipping_prob=type('', (object,), {'rvs': lambda _: 0})(),
                                               clipping_prob_args={})
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_array_almost_equal(input_signal, output_signal, 1)

    def test_AudioVolumePerturbator_with_volume_change(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioVolumePerturbator(p=1.0,
                                             volume_prob=type('', (object,), {'rvs': lambda _: 10})(),
                                             volume_prob_args={})
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_signal, output_signal)

    def test_AudioVolumePerturbator_without_volume_change(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioVolumePerturbator(p=1.0,
                                             volume_prob=type('', (object,), {'rvs': lambda _: 0})(),
                                             volume_prob_args={})
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_array_almost_equal(input_signal, output_signal, 4)

    def test_AudioEchoPerturbator_with_echo(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioEchoPerturbator(p=1.0,
                                           echo_prob=type('', (object,), {'rvs': lambda _: 1.0})(),
                                           echo_prob_args={})
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_signal, output_signal)

    def test_AudioEchoPerturbator_without_echo(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioEchoPerturbator(p=1.0,
                                           echo_prob=type('', (object,), {'rvs': lambda _: 0.0})(),
                                           echo_prob_args={})
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_array_equal(output_signal, input_signal * 2)

    def test_AudioBackgroundNoisePerturbator_with_noise(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioBackgroundNoisePerturbator(p=1.0, noise_types=[''])
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_signal, output_signal)

    def test_AudioShortNoisePerturbator_with_noise(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioShortNoisePerturbator(p=1.0, noise_types=[''])
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_signal, output_signal)

    def test_AudioImpulseResponsePerturbator_with_noise(self):
        input_signal = self._generate_random_audio()
        perturbator = AudioImpulseResponsePerturbator(p=1.0, impulse_types=[''])
        output_signal,sample_rate = perturbator.apply((input_signal, 44100))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_signal, output_signal)


if __name__ == '__main__':
    unittest.main()
