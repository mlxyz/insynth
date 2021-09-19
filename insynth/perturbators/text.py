import random
import re

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

    def apply(self, original_input: str):
        new_text = original_input
        for correct_word, misspellings in self.misspell_map.items():
            new_text = re.sub('(?<!\w)' + re.escape(correct_word) + '(?=\W|$)',
                              lambda match: random.choice(
                                  misspellings) if random.random() < self.typo_rate else match.group(0),
                              new_text, flags=re.IGNORECASE)
        return new_text


class TextCasePerturbator(BlackboxTextPerturbator):
    def __init__(self, probability=0.2):
        self.probability = probability

    def apply(self, original_input: str, probability=0.2):
        return ''.join((x.lower() if x.isupper() else x.upper()) if random.random() < probability else x for x in
                       original_input)


class TextWordRemovalPerturbator(BlackboxTextPerturbator):
    def __init__(self, probability=0.2):
        self.probability = probability

    def apply(self, original_input: str):
        return re.sub('(?<!\w)\w+(?=\W|$)',
                      lambda match: '' if random.random() < self.probability else match.group(0),
                      original_input, flags=re.IGNORECASE)


class TextStopWordRemovalPerturbator(BlackboxTextPerturbator):
    def __init__(self, probability=0.2):
        self.probability = probability

    def apply(self, original_input: str):
        new_text = original_input
        for stop_word in STOP_WORDS:
            new_text = re.sub('(?<!\w)' + re.escape(stop_word) + '(?=\W|$)',
                              lambda match: '' if random.random() < self.probability else match.group(0),
                              original_input, flags=re.IGNORECASE)
        return new_text


class TextWordSwitchPerturbator(BlackboxTextPerturbator):
    def __init__(self, probability=0.2):
        self.probability = probability
        self.was_switched = False

    def apply(self, original_input):
        tokens = re.findall('(?<!\w)\w+(?=\W|$)', original_input, flags=re.IGNORECASE)

        return re.sub('(?<!\w)\w+(?=\W|$)',
                      lambda match: self.switch_word(match, tokens),
                      original_input, flags=re.IGNORECASE)

    def switch_word(self, match, tokens: list):
        if self.was_switched:
            ret_val = tokens.pop(0)
            self.was_switched = False
            return ret_val
        if len(tokens) <= 2:
            return match.group(0)
        if random.random() < self.probability:
            tokens.pop(0)
            ret_val = tokens[0]
            tokens[0] = match.group(0)
            self.was_switched = True
        else:
            ret_val = tokens.pop(0)
        return ret_val


class TextCharacterSwitchPerturbator(BlackboxTextPerturbator):
    def __init__(self, probability=0.2):
        self.probability = probability
        self.was_switched = False

    def apply(self, original_input):
        return re.sub('(?<!\w)\w+(?=\W|$)',
                      lambda match: self.create_word_with_characters_switched(match),
                      original_input, flags=re.IGNORECASE)

    def create_word_with_characters_switched(self, match):

        text = match.group(0)
        tokens = re.findall('\w', text, flags=re.IGNORECASE)

        return re.sub('\w',
                      lambda match: self.switch_characters(match, tokens),
                      text, flags=re.IGNORECASE)

    def switch_characters(self, match, tokens):
        if self.was_switched:
            ret_val = tokens.pop(0)
            self.was_switched = False
            return ret_val
        if len(tokens) <= 2:
            return match.group(0)
        if random.random() < self.probability:
            tokens.pop(0)
            ret_val = tokens[0]
            tokens[0] = match.group(0)
            self.was_switched = True
        else:
            ret_val = tokens.pop(0)
        return ret_val
