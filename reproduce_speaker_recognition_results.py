import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from insynth.runners.runner import ComprehensiveAudioRunner

logging.basicConfig(level=logging.INFO)
DATASET_ROOT = os.path.join("data/speaker_recognition/16000_pcm_speeches")

AUDIO_SUBFOLDER = "audio"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)

VALID_SPLIT = 0.1

SHUFFLE_SEED = 43

SAMPLING_RATE = 16000

BATCH_SIZE = 128
EPOCHS = 100


def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(
        lambda x: path_to_audio(x), num_parallel_calls=tf.data.AUTOTUNE
    )
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio, sr


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    if audio.shape[1] > 16000:
        audio = audio[:, :16000]
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])
    # return tf.math.abs(fft[:, : 8000, :])


# Get the list of audio file paths along with their corresponding labels

class_names = os.listdir(DATASET_AUDIO_PATH)
print("Our class names: {}".format(class_names, ))

audio_paths = []
labels = []
for label, name in enumerate(class_names):
    print("Processing speaker {}".format(name, ))
    dir_path = Path(DATASET_AUDIO_PATH) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

print(
    "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
)

# Shuffle
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# Split into training and validation
num_val_samples = int(VALID_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]

print("Using {} files for validation.".format(num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

# Create 2 datasets, one for training and the other for validation


with open('data/speaker_recognition/train_paths.txt', 'r') as f:
    train_audio_paths = f.read().splitlines()
with open('data/speaker_recognition/train_labels.txt', 'r') as f:
    train_labels = list(map(int, f.read().splitlines()))
with open('data/speaker_recognition/valid_paths.txt', 'r') as f:
    valid_audio_paths = f.read().splitlines()
with open('data/speaker_recognition/valid_labels.txt', 'r') as f:
    valid_labels = list(map(int, f.read().splitlines()))

train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
#train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED)

valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
#valid_ds = valid_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED)

# Transform audio wave to the frequency domain using `audio_to_fft`
# train_ds = train_ds.map(
#    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
# )
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# valid_ds = valid_ds.map(
#    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
# )
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)


def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)


model = build_model((SAMPLING_RATE // 2, 1), len(class_names))
model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.load_weights('data/speaker_recognition/model_weights.h5')

x_test_data_generator = lambda: ((np.squeeze(sample[0][0]), sample[0][1].numpy()) for sample in valid_ds)
x_snac_data_generator = lambda: ((np.squeeze(sample[0][0]), sample[0][1].numpy()) for sample in train_ds.take(10))

print(valid_labels)


def pre_predict(sample):
    audio = sample[0]
    audio = np.expand_dims(audio, axis=0)
    audio = np.expand_dims(audio, axis=-1)
    return audio_to_fft(audio)


runner = ComprehensiveAudioRunner(x_test_data_generator, valid_labels, model, x_snac_data_generator,
                                  pre_predict_lambda=pre_predict)
report, robustness = runner.run()

print(report.to_string())
print(robustness)
