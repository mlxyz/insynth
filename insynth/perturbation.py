from abc import ABC, abstractmethod

from insynth.input import AbstractInput, AudioInput, ImageInput, TextInput


class AbstractBlackboxPerturbator(ABC):
    @abstractmethod
    def apply(self, original_input: AbstractInput):
        pass


class BlackboxImagePerturbator(AbstractBlackboxPerturbator):
    @abstractmethod
    def apply(self, original_input: ImageInput):
        pass


class BlackboxAudioPerturbator(AbstractBlackboxPerturbator):
    @abstractmethod
    def apply(self, original_input: AudioInput):
        pass


class BlackboxTextPerturbator(AbstractBlackboxPerturbator):
    @abstractmethod
    def apply(self, original_input: TextInput):
        pass


class AbstractWhiteboxPerturbator(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def apply(self, original_input: AbstractInput):
        pass


class WhiteboxImagePerturbator(AbstractWhiteboxPerturbator):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def apply(self, original_input: ImageInput):
        pass


class WhiteboxAudioPerturbator(AbstractWhiteboxPerturbator):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def apply(self, original_input: AudioInput):
        pass


class WhiteboxTextPerturbator(AbstractWhiteboxPerturbator):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def apply(self, original_input: TextInput):
        pass
