from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report


class AbstractRunner(ABC):
    @abstractmethod
    def run(self):
        pass


class BasicRunner(AbstractRunner):

    def __init__(self, perturbators, coverage_calculators, dataset_x, dataset_y, model):
        self.perturbators = perturbators
        self.coverage_calculators = coverage_calculators
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.model = model

    def run(self, save_images=False):
        results = {}
        y_pred = np.argmax(self.model.predict(self.dataset_x, verbose=1), axis=1)
        self.put_results_into_dict(results, 'original', self.dataset_y, y_pred)
        for index1, perturbator in enumerate(self.perturbators):
            perturbator_name = type(perturbator).__name__
            perturbated_dataset = []
            for index2, sample in enumerate(self.dataset_x):
                new_image = perturbator.apply(Image.fromarray(sample))
                if save_images:
                    new_image.save(f'data/images/{perturbator_name}_{index2}.jpg', 'JPEG')
                perturbated_dataset.append(np.array(new_image))
            predictions = np.argmax(self.model.predict(np.array(perturbated_dataset), verbose=1), axis=1)
            self.put_results_into_dict(results, perturbator_name, self.dataset_y, predictions)
        return pd.DataFrame.from_dict(results, orient='index')

    def put_results_into_dict(self, dct, name, y_true, y_pred):
        results = classification_report(y_true,
                                        y_pred,
                                        output_dict=True)
        dct[name] = {
            'acc': results['accuracy'],
            'macro_f1': results['macro avg']['f1-score'],
            'macro_rec': results['macro avg']['recall'],
            'macro_prec': results['macro avg']['precision'],
            'micro_f1': results['weighted avg']['f1-score'],
            'micro_rec': results['weighted avg']['recall'],
            'micro_prec': results['weighted avg']['precision']}
