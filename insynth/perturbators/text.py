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

import random
import re

from scipy.stats import norm

from insynth.data import utils
from insynth.perturbation import BlackboxTextPerturbator

STOP_WORDS = ['i', 'me', 'and', 'an']


class TextTypoPerturbator(BlackboxTextPerturbator):

    def __init__(self, p=0.5, typo_prob=norm, typo_prob_args={'loc': 0.2, 'scale': 0.1}):
        super().__init__(p)
        self.typo_prob = typo_prob
        self.typo_prob_args = typo_prob_args
        utils.download_and_unzip(
            'https://insynth-data.s3.eu-central-1.amazonaws.com/misspellings.zip',
            'data/text/misspellings/')
        with open('data/text/misspellings/misspellings.dat') as f:
            self.misspell_map = {}
            correct_word = None
            for line in f.read().splitlines():
                if line.startswith('$'):
                    correct_word = line[1:].lower()
                    self.misspell_map[correct_word] = []
                else:
                    self.misspell_map[correct_word].append(line.lower())

    def _internal_apply(self, original_input: str):
        typo_rate = self.typo_prob.rvs(**self.typo_prob_args)
        new_text = original_input
        for correct_word, misspellings in self.misspell_map.items():
            new_text = re.sub('(?<!\w)' + re.escape(correct_word) + '(?=\W|$)',
                              lambda match: random.choice(
                                  misspellings) if random.random() < typo_rate else match.group(0),
                              new_text, flags=re.IGNORECASE)
        return new_text


class TextCasePerturbator(BlackboxTextPerturbator):
    def __init__(self, p=0.5, case_switch_prob=norm, case_switch_prob_args={'loc': 0.2, 'scale': 0.1}):
        super().__init__(p)
        self.case_switch_prob = case_switch_prob
        self.case_switch_prob_args = case_switch_prob_args

    def _internal_apply(self, original_input: str):
        case_switch_prob = self.case_switch_prob.rvs(**self.case_switch_prob_args)
        return ''.join((x.lower() if x.isupper() else x.upper()) if random.random() < case_switch_prob else x for x in
                       original_input)


class TextWordRemovalPerturbator(BlackboxTextPerturbator):
    def __init__(self, p=0.5, word_removal_prob=norm, word_removal_prob_args={'loc': 0.2, 'scale': 0.1}):
        super().__init__(p)
        self.word_removal_prob = word_removal_prob
        self.word_removal_prob_args = word_removal_prob_args

    def _internal_apply(self, original_input: str):
        word_removal_prob = self.word_removal_prob.rvs(**self.word_removal_prob_args)
        return re.sub('(?<!\w)\w+(?=\W|$)',
                      lambda match: '' if random.random() < word_removal_prob else match.group(0),
                      original_input, flags=re.IGNORECASE)


class TextStopWordRemovalPerturbator(BlackboxTextPerturbator):
    def __init__(self, p=0.5, stop_word_removal_prob=norm, stop_word_removal_prob_args={'loc': 0.2, 'scale': 0.1}):
        super().__init__(p)
        self.stop_word_removal_prob = stop_word_removal_prob
        self.stop_word_removal_prob_args = stop_word_removal_prob_args

    def _internal_apply(self, original_input: str):
        stop_word_removal_prob = self.stop_word_removal_prob.rvs(**self.stop_word_removal_prob_args)
        new_text = original_input
        for stop_word in STOP_WORDS:
            new_text = re.sub('(?<!\w)' + re.escape(stop_word) + '(?=\W|$)',
                              lambda match: '' if random.random() < stop_word_removal_prob else match.group(0),
                              original_input, flags=re.IGNORECASE)
        return new_text


class TextWordSwitchPerturbator(BlackboxTextPerturbator):
    def __init__(self, p=0.5, word_switch_prob=norm, word_switch_prob_args={'loc': 0.2, 'scale': 0.1}):
        super().__init__(p)
        self.word_switch_prob = word_switch_prob
        self.word_switch_prob_args = word_switch_prob_args
        self._was_switched = False

    def _internal_apply(self, original_input):
        self._was_switched = False
        switch_word_prob = self.word_switch_prob.rvs(**self.word_switch_prob_args)
        tokens = re.findall('(?<!\w)\w+(?=\W|$)', original_input, flags=re.IGNORECASE)

        return re.sub('(?<!\w)\w+(?=\W|$)',
                      lambda match: self.switch_word(match, tokens, switch_word_prob),
                      original_input, flags=re.IGNORECASE)

    def switch_word(self, match, tokens: list, prob):
        if self._was_switched:
            ret_val = tokens.pop(0)
            self._was_switched = False
            return ret_val
        if len(tokens) <= 2:
            return match.group(0)
        if random.random() < prob:
            tokens.pop(0)
            ret_val = tokens[0]
            tokens[0] = match.group(0)
            self._was_switched = True
        else:
            ret_val = tokens.pop(0)
        return ret_val


class TextCharacterSwitchPerturbator(BlackboxTextPerturbator):
    def __init__(self, p=0.5, char_switch_prob=norm, char_switch_prob_args={'loc': 0.2, 'scale': 0.1}):
        super().__init__(p)
        self._was_switched = False
        self.char_switch_prob = char_switch_prob
        self.char_switch_prob_args = char_switch_prob_args

    def _internal_apply(self, original_input):
        self._was_switched = False
        char_switch_prob = self.char_switch_prob.rvs(**self.char_switch_prob_args)
        return re.sub('(?<!\w)\w+(?=\W|$)',
                      lambda match: self.create_word_with_characters_switched(match, char_switch_prob),
                      original_input, flags=re.IGNORECASE)

    def create_word_with_characters_switched(self, match, prob):

        text = match.group(0)
        tokens = re.findall('\w', text, flags=re.IGNORECASE)

        return re.sub('\w',
                      lambda match: self.switch_characters(match, tokens, prob),
                      text, flags=re.IGNORECASE)

    def switch_characters(self, match, tokens, prob):
        if self._was_switched:
            ret_val = tokens.pop(0)
            self._was_switched = False
            return ret_val
        if len(tokens) <= 2:
            return match.group(0)
        if random.random() < prob:
            tokens.pop(0)
            ret_val = tokens[0]
            tokens[0] = match.group(0)
            self._was_switched = True
        else:
            ret_val = tokens.pop(0)
        return ret_val


class TextPunctuationErrorPerturbator(BlackboxTextPerturbator):

    def __init__(self, p=0.5, punct_error_prob=norm, punct_error_prob_args={'loc': 0.2, 'scale': 0.1}):
        super().__init__(p)
        self.punct_error_prob = punct_error_prob
        self.punct_error_prob_args = punct_error_prob_args

    def _internal_apply(self, original_input):
        punct_error_prob = self.punct_error_prob.rvs(**self.punct_error_prob_args)
        original_input = self.apply_apostrophe_error(original_input, punct_error_prob)
        original_input = self.apply_period_error(original_input, punct_error_prob)
        original_input = self.apply_comma_error(original_input, punct_error_prob)
        original_input = self.apply_hyphen_error(original_input, punct_error_prob)
        original_input = self.apply_common_errors(original_input, punct_error_prob)
        return original_input

    def apply_apostrophe_error(self, text_input, prob):
        return re.sub('(?<!\w)\w{3,}s(?=\W|$)',
                      lambda match: match.group(0)[:-1] + '\'' + match.group(0)[-1:]
                      if random.random() < prob else match.group(0),
                      text_input, flags=re.IGNORECASE)

    def apply_period_error(self, text_input, prob):
        return re.sub('\.',
                      lambda match: random.choice([',', ';'])
                      if random.random() < prob else match.group(0),
                      text_input, flags=re.IGNORECASE)

    def apply_comma_error(self, text_input, prob):
        return re.sub(',',
                      lambda match: random.choice(['', ';'])
                      if random.random() < prob else match.group(0),
                      text_input, flags=re.IGNORECASE)

    def apply_common_errors(self, text_input, prob):
        common_error_words = [
            'they\'re',
            'you\'re',
            'it\'s',
            'he\'s',
        ]
        joined_common_error_words = '|'.join(common_error_words)
        return re.sub(f'(?<!\w)({joined_common_error_words})(?=\W|$)',
                      lambda match: match.group(0).replace('\'', '')
                      if random.random() < prob else match.group(0),
                      text_input, flags=re.IGNORECASE)

    def apply_hyphen_error(self, text_input, prob):
        return re.sub('-|–',
                      lambda match: '–' if match.group(0) == '-' else '–'
                      if random.random() < prob else match.group(0),
                      text_input, flags=re.IGNORECASE)
