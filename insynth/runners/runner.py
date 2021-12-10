from abc import ABC, abstractmethod

import numpy as np
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

    def run(self):
        
        y_pred = np.argmax(self.model.predict(self.dataset_x, verbose=1), axis=1)
        original_class_report = classification_report(self.dataset_y, y_pred, output_dict=True)
        mutated_dataset = []
        for index1, perturbator in enumerate(self.perturbators):
            for index2, sample in enumerate(self.dataset_x):
                new_image = perturbator.apply(Image.fromarray(sample))
                new_image.save(f'data/images/{index1}_{index2}.jpg', 'JPEG')
                mutated_dataset.append(np.array(new_image))

        mutated_dataset = np.array(mutated_dataset)
        combined_dataset = mutated_dataset  # np.concatenate((self.dataset_x, mutated_dataset))
        predictions = self.model.predict(combined_dataset, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        mutated_class_report = classification_report(np.tile(self.dataset_y, len(self.perturbators)),
                                                     y_pred,
                                                     output_dict=True)
        print('ORIGINAL')
        print(original_class_report)

        print()

        print('MUTATED')
        print(mutated_class_report)
