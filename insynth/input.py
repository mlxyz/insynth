#  Copyright (c) 2022, Chair of Software Technology
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#  - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  - Neither the name of the University Mannheim nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
