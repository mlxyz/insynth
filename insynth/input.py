from abc import ABC
from io import BytesIO

import librosa
import requests
from PIL import Image


class AbstractInput(ABC):
    pass


class ImageInput(AbstractInput):
    def __init__(self, image: Image):
        self.image = image

    @staticmethod
    def from_file(file_path: str):
        return ImageInput(Image.open(file_path))

    @staticmethod
    def from_url(file_url: str):
        return ImageInput(Image.open(requests.get(file_url, stream=True).raw))

    @staticmethod
    def from_bytes(bytes):
        return ImageInput(Image.open(BytesIO(bytes)))


class AudioInput(AbstractInput):
    def __init__(self, signal, sr):
        self.signal = signal
        self.sr = sr

    @staticmethod
    def from_file(file_name):
        return AudioInput(*librosa.load(file_name, sr=None))


class TextInput(AbstractInput):
    def __init__(self, text: str):
        self.text = text

    @staticmethod
    def from_file(file_path: str):
        return TextInput(open(file_path).read())

    @staticmethod
    def from_url(file_url: str):
        return TextInput(requests.get(file_url).text)

    @staticmethod
    def from_bytes(bytes):
        return TextInput(bytes.decode('utf-8'))

    def get_text(self) -> str:
        return self.text
