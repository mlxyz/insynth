import random
import re

from insynth.input import TextInput
from insynth.perturbation import BlackboxTextPerturbator

STOP_WORDS = ['i', 'me', 'and']


class TextTypoPerturbator(BlackboxTextPerturbator):

    def __init__(self, typo_rate=0.2):
        self.typo_rate = typo_rate
        with open('data/audio/missp.dat') as f:
            self.misspell_map = {}
            correct_word = None
            for line in f.read().splitlines():
                if line.startswith('$'):
                    correct_word = line[1:].lower()
                    self.misspell_map[correct_word] = []
                else:
                    self.misspell_map[correct_word].append(line.lower())

    def apply(self, original_input: TextInput):
        new_text = original_input.text
        for correct_word, misspellings in self.misspell_map.items():
            new_text = re.sub('(?<!\w)' + re.escape(correct_word) + '(?=\W|$)',
                              lambda match: random.choice(
                                  misspellings) if random.random() < self.typo_rate else match.group(0),
                              new_text, flags=re.IGNORECASE)
        return new_text


class TextCasePerturbator(BlackboxTextPerturbator):
    def __init__(self, probability=0.2):
        self.probability = probability

    def apply(self, original_input: TextInput, probability=0.2):
        return ''.join((x.lower() if x.isupper() else x.upper()) if random.random() < probability else x for x in
                       original_input.text)


class TextWordRemovalPerturbator(BlackboxTextPerturbator):
    def __init__(self, probability=0.2):
        self.probability = probability

    def apply(self, original_input: TextInput):
        return re.sub('(?<!\w)\w+(?=\W|$)',
                      lambda match: '' if random.random() < self.probability else match.group(0),
                      original_input.text, flags=re.IGNORECASE)


class TextStopWordRemovalPerturbator(BlackboxTextPerturbator):
    def __init__(self, probability=0.2):
        self.probability = probability

    def apply(self, original_input: TextInput):
        new_text = original_input.text
        for stop_word in STOP_WORDS:
            new_text = re.sub('(?<!\w)' + re.escape(stop_word) + '(?=\W|$)',
                              lambda match: '' if random.random() < self.probability else match.group(0),
                              original_input.text, flags=re.IGNORECASE)
        return new_text
