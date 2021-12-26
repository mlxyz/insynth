import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm

from experiments.old_coverage_calculators import NeuronCoverageCalculator as OldNeuronCoverageCalculator, \
    StrongNeuronActivationCoverageCalculator as OldStrongNeuronActivationCoverageCalculator, \
    KMultiSectionNeuronCoverageCalculator as OldKMultiSectionNeuronCoverageCalculator, \
    NeuronBoundaryCoverageCalculator as OldNeuronBoundaryCoverageCalculator, \
    TopKNeuronCoverageCalculator as OldTopKNeuronCoverageCalculator, \
    TopKNeuronPatternsCalculator as OldTopKNeuronPatternsCalculator
from insynth.metrics.coverage.neuron import NeuronCoverageCalculator, StrongNeuronActivationCoverageCalculator, \
    KMultiSectionNeuronCoverageCalculator, NeuronBoundaryCoverageCalculator, TopKNeuronCoverageCalculator, \
    TopKNeuronPatternsCalculator

xception_model = tf.keras.applications.Xception()
ds = tfds.load('imagenet_v2', split='test', shuffle_files=True)

val_ds = lambda: (np.expand_dims(np.array(Image.fromarray(sample['image'].numpy()).resize((299, 299)).convert('RGB')),
                                 axis=0) for sample in ds.take(1000))
snac_ds = lambda: (np.expand_dims(np.array(Image.fromarray(sample['image'].numpy()).resize((299, 299)).convert('RGB')),
                                  axis=0) for sample in ds.skip(1000).take(1000))

old_coverage_calculators = [
    OldNeuronCoverageCalculator(xception_model),
    OldStrongNeuronActivationCoverageCalculator(xception_model),
    OldKMultiSectionNeuronCoverageCalculator(xception_model),
    OldNeuronBoundaryCoverageCalculator(xception_model),
    OldTopKNeuronCoverageCalculator(xception_model),
    OldTopKNeuronPatternsCalculator(xception_model)
]

new_coverage_calculators = [
    NeuronCoverageCalculator(xception_model),
    StrongNeuronActivationCoverageCalculator(xception_model),
    KMultiSectionNeuronCoverageCalculator(xception_model),
    NeuronBoundaryCoverageCalculator(xception_model),
    TopKNeuronCoverageCalculator(xception_model),
    TopKNeuronPatternsCalculator(xception_model)
]
results = {}
for old_calc, new_calc in zip(old_coverage_calculators, new_coverage_calculators):
    calc_name = type(old_calc).__name__
    print(f'Comparing {type(old_calc).__name__} with {type(new_calc).__name__}...')
    results[calc_name] = {}
    update_neuron_bounds_op = getattr(old_calc, "update_neuron_bounds", None)
    if callable(update_neuron_bounds_op):
        print('Calculator requires neuron bounds.')
        start_time = time.time()
        for sample in tqdm(snac_ds(), desc='[Old Calculator] Processing SNAC...'):
            old_calc.update_neuron_bounds(sample)
        end_time = time.time()
        print(f'SNAC Processing of Old Calculator Done: took {end_time - start_time} seconds.')
        results[calc_name]['snac_old'] = end_time - start_time
        start_time = time.time()
        for sample in tqdm(snac_ds(), desc='[New Calculator] Processing SNAC...'):
            new_calc.update_neuron_bounds(sample)
        print(f'SNAC Processing of New Calculator Done: took {end_time - start_time} seconds.')
        results[calc_name]['snac_new'] = end_time - start_time
        end_time = time.time()
    start_time = time.time()
    for sample in tqdm(val_ds(), desc='[Old Calculator] Processing Samples...'):
        old_calc.update_coverage(sample)
    end_time = time.time()
    print(f'Old Calculator Done: took {end_time - start_time} seconds.')
    results[calc_name]['old'] = end_time - start_time

    start_time = time.time()
    for sample in tqdm(val_ds(), desc='[New Calculator] Processing Samples...'):
        new_calc.update_coverage(sample)
    end_time = time.time()
    print(f'New Calculator Done: took {end_time - start_time} seconds.')
    results[calc_name]['new'] = end_time - start_time

df = pd.DataFrame.from_dict(results, orient='index')

df.to_csv('performance_comparison.csv')
