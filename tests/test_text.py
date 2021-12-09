import unittest

from insynth.perturbators.text import TextTypoPerturbator, TextCasePerturbator, TextWordRemovalPerturbator, \
    TextStopWordRemovalPerturbator, TextWordSwitchPerturbator, TextCharacterSwitchPerturbator, \
    TextPunctuationErrorPerturbator


class TestText(unittest.TestCase):
    def test_TextTypoPerturbator_with_typos(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextTypoPerturbator(p=1.0,
                                          typo_prob=type('', (object,), {'rvs': lambda _: 1.0})(),
                                          typo_prob_args={})

        output_text = perturbator.apply(input_text)

        assert not output_text == input_text

    def test_TextTypoPerturbator_without_typos(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextTypoPerturbator(p=1.0,
                                          typo_prob=type('', (object,), {'rvs': lambda _: 0.0})(),
                                          typo_prob_args={})

        output_text = perturbator.apply(input_text)

        assert output_text == input_text

    def test_TextCasePerturbator_with_case_switch(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextCasePerturbator(p=1.0,
                                          case_switch_prob=type('', (object,), {'rvs': lambda _: 1.0})(),
                                          case_switch_prob_args={})

        output_text = perturbator.apply(input_text)

        assert output_text == 'tHIS IS AN EXAMPLE TEXT WHICH IS USED FOR TESTING THE FUNCTIONALITY OF THE INSYNTH LIBRARY.'

    def test_TextCasePerturbator_without_case_switch(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextCasePerturbator(p=1.0,
                                          case_switch_prob=type('', (object,), {'rvs': lambda _: 0.0})(),
                                          case_switch_prob_args={})

        output_text = perturbator.apply(input_text)

        assert output_text == input_text

    def test_TextWordRemovalPerturbator_with_word_removal(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextWordRemovalPerturbator(p=1.0,
                                                 word_removal_prob=type('', (object,), {'rvs': lambda _: 1.0})(),
                                                 word_removal_prob_args={})

        output_text = perturbator.apply(input_text)

        assert output_text.lstrip() == '.'

    def test_TextWordRemovalPerturbator_without_word_removal(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextWordRemovalPerturbator(p=1.0,
                                                 word_removal_prob=type('', (object,), {'rvs': lambda _: 0.0})(),
                                                 word_removal_prob_args={})

        output_text = perturbator.apply(input_text)

        assert output_text == input_text

    def test_TextStopWordRemovalPerturbator_with_stopword_removal(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextStopWordRemovalPerturbator(p=1.0,
                                                     stop_word_removal_prob=type('', (object,),
                                                                                 {'rvs': lambda _: 1.0})(),
                                                     stop_word_removal_prob_args={})

        output_text = perturbator.apply(input_text)
        # TODO: fix this
        assert not output_text == input_text

    def test_TextStopWordRemovalPerturbator_without_stopword_removal(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextStopWordRemovalPerturbator(p=1.0,
                                                     stop_word_removal_prob=type('', (object,),
                                                                                 {'rvs': lambda _: 0.0})(),
                                                     stop_word_removal_prob_args={})

        output_text = perturbator.apply(input_text)
        assert output_text == input_text

    def test_TextWordSwitchPerturbator_with_word_switch(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextWordSwitchPerturbator(p=1.0,
                                                word_switch_prob=type('', (object,),
                                                                      {'rvs': lambda _: 1.0})(),
                                                word_switch_prob_args={})

        output_text = perturbator.apply(input_text)

        assert output_text == 'is This example an which text used is testing for functionality the the of insynth library.'

    def test_TextWordSwitchPerturbator_without_word_switch(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextWordSwitchPerturbator(p=1.0,
                                                word_switch_prob=type('', (object,),
                                                                      {'rvs': lambda _: 0.0})(),
                                                word_switch_prob_args={})

        output_text = perturbator.apply(input_text)

        assert output_text == input_text

    def test_TextCharacterSwitchPerturbator_with_char_switch(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextCharacterSwitchPerturbator(p=1.0,
                                                     char_switch_prob=type('', (object,),
                                                                           {'rvs': lambda _: 1.0})(),
                                                     char_switch_prob_args={})

        output_text = perturbator.apply(input_text)

        assert output_text == 'hTis is an xemalpe etxt hwcih is sued ofr ettsnig hte ufcnitnolatiy of hte niystnh ilrbray.'

    def test_TextCharacterSwitchPerturbator_without_char_switch(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextCharacterSwitchPerturbator(p=1.0,
                                                     char_switch_prob=type('', (object,),
                                                                           {'rvs': lambda _: 0.0})(),
                                                     char_switch_prob_args={})

        output_text = perturbator.apply(input_text)

        assert output_text == input_text

    def test_TextPunctuationErrorPerturbator_with_punct_error(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextPunctuationErrorPerturbator(p=1.0,
                                                      punct_error_prob=type('', (object,),
                                                                            {'rvs': lambda _: 1.0})(),
                                                      punct_error_prob_args={})

        output_text = perturbator.apply(input_text)
        print(output_text)
        assert output_text == 'Thi\'s is an example text which is used for testing the functionality of the insynth library;' \
               or output_text == 'Thi\'s is an example text which is used for testing the functionality of the insynth library,' \
               or output_text == 'Thi\'s is an example text which is used for testing the functionality of the insynth library'

    def test_TextPunctuationErrorPerturbator_without_punct_error(self):
        input_text = 'This is an example text which is used for testing the functionality of the insynth library.'

        perturbator = TextPunctuationErrorPerturbator(p=1.0,
                                                      punct_error_prob=type('', (object,),
                                                                            {'rvs': lambda _: 0.0})(),
                                                      punct_error_prob_args={})

        output_text = perturbator.apply(input_text)

        assert output_text == input_text


if __name__ == '__main__':
    unittest.main()
