from insynth.perturbators.audio import AudioImpulseResponsePerturbator
import librosa
from scipy.io.wavfile import write
perturbator = AudioImpulseResponsePerturbator(impulse_types=['echo_thief', 'mit_mcdermott'])
signal, sr=librosa.load("E:\\Daten\\Downloads\\male.wav", sr=None)
result = perturbator.apply((signal, sr))
write("output.wav", sr, result)