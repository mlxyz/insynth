import os
import pickle
import re
import string

import numpy as np
import tensorflow as tf
# download and extract https://insynth-data.s3.eu-central-1.amazonaws.com/sentiment_analysis_experiment.zip to data/
from keras.layers import TextVectorization

from insynth.perturbators.text import TextTypoPerturbator, TextCasePerturbator, TextWordRemovalPerturbator, \
    TextStopWordRemovalPerturbator, TextWordSwitchPerturbator, TextCharacterSwitchPerturbator, \
    TextPunctuationErrorPerturbator
from insynth.runners.runner import BasicTextRunner

DATA_ROOT_PATH = '../data/sentiment_analysis/'
batch_size = 32

raw_val_ds_x = tf.keras.preprocessing.text_dataset_from_directory(
    DATA_ROOT_PATH + "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    shuffle=True
).unbatch()
raw_val_ds_y = tf.keras.preprocessing.text_dataset_from_directory(
    DATA_ROOT_PATH + "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    shuffle=True
).unbatch()
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    DATA_ROOT_PATH + "aclImdb/test", batch_size=batch_size, shuffle=True
).unbatch()

print(f"Number of batches in raw_val_ds: {raw_val_ds_x.cardinality()}")
print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


model = tf.keras.models.load_model(DATA_ROOT_PATH + 'model.h5')
# bug see https://stackoverflow.com/questions/70255845/tensorflow-textvectorization-producing-ragged-tensor-with-no-padding-after-loadi
from_disk = pickle.load(open(DATA_ROOT_PATH + "tv_layer.pkl", "rb"))
text_vectroize_layer = TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                         output_mode='int',
                                         output_sequence_length=from_disk['config']['output_sequence_length'])

text_vectroize_layer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
text_vectroize_layer.set_weights(from_disk['weights'])

perturbators = [
    *[TextTypoPerturbator(p=1.0, typo_prob=type('', (object,), {'rvs': lambda _, **kwargs: kwargs['value']})(),
                          typo_prob_args={'value': typo}) for typo in np.linspace(0.01, 1.0, 30)],
    *[TextCasePerturbator(p=1.0, case_switch_prob=type('', (object,), {'rvs': lambda _, **kwargs: kwargs['value']})(),
                          case_switch_prob_args={'value': case_switch}) for case_switch in np.linspace(0.01, 1.0, 30)],
    *[TextWordRemovalPerturbator(p=1.0,
                                 word_removal_prob=type('', (object,), {'rvs': lambda _, **kwargs: kwargs['value']})(),
                                 word_removal_prob_args={'value': word_removal}) for word_removal in
      np.linspace(0.01, 1.0, 30)],
    *[TextStopWordRemovalPerturbator(p=1.0, stop_word_removal_prob=type('', (object,),
                                                                        {'rvs': lambda _, **kwargs: kwargs['value']})(),
                                     stop_word_removal_prob_args={'value': stop_word_removal}) for stop_word_removal in
      np.linspace(0.01, 1.0, 30)],
    *[TextWordSwitchPerturbator(p=1.0,
                                word_switch_prob=type('', (object,), {'rvs': lambda _, **kwargs: kwargs['value']})(),
                                word_switch_prob_args={'value': word_switch}) for word_switch in
      np.linspace(0.01, 1.0, 30)],
    *[TextCharacterSwitchPerturbator(p=1.0, char_switch_prob=type('', (object,),
                                                                  {'rvs': lambda _, **kwargs: kwargs['value']})(),
                                     char_switch_prob_args={'value': char_switch}) for char_switch in
      np.linspace(0.01, 1.0, 30)],
    *[TextPunctuationErrorPerturbator(p=1.0, punct_error_prob=type('', (object,),
                                                                   {'rvs': lambda _, **kwargs: kwargs['value']})(),
                                      punct_error_prob_args={'value': punct_error}) for punct_error in
      np.linspace(0.01, 1.0, 30)]]


def vectorize_text(text):
    return text_vectroize_layer([text])


x_test_data = [sample[0].numpy().decode('utf-8') for sample in raw_val_ds_x]
x_test_data_generator = lambda: x_test_data
y_test = [sample[1].numpy() for sample in raw_val_ds_y]
model.summary()
runner = BasicTextRunner(perturbators, [], x_test_data_generator, y_test, model,
                         pre_predict_lambda=vectorize_text)

os.makedirs('output/sentiment_analysis/')
report, robustness = runner.run(True, 'output/sentiment_analysis/')

print(report.to_string())
print(robustness)

report.to_csv('output/sentiment_analysis/report.csv')
