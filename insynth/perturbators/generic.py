import warnings

import numpy as np
import scipy.stats as st

from insynth.perturbation import AbstractBlackboxPerturbator


class GenericPerturbator(AbstractBlackboxPerturbator):
    DISTRIBUTIONS_TO_TEST = [
        'norm',
        'chi2',
        'alpha',
        'cosine',
        'beta',
        'f',
        'laplace',
        'logistic',
        'pearson3',
        'trapezoid'
    ]

    def _internal_apply(self, original_input):
        return [distribution[0].rvs(*distribution[1]) for distribution in self.distributions]

    def __init__(self, p=0.5):
        super().__init__(p)
        self.distributions = []

    def fit(self, dataset):
        self.distributions = [self.best_fit_distribution(column) for column in dataset.transpose()]

    def best_fit_distribution(self, data, bins=200):
        """Model data by finding best fit distribution to data"""

        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        best_distributions = []

        for index, distribution in enumerate(
                [d for d in self.DISTRIBUTIONS_TO_TEST]):

            distribution = getattr(st, distribution)
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    params = distribution.fit(data)

                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    prob_dist_func = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sum_squared_errors = np.sum(np.power(y - prob_dist_func, 2.0))

                    best_distributions.append((distribution, params, sum_squared_errors))

            except:
                pass
        # return distribution with lowest error
        return sorted(best_distributions, key=lambda dist: dist[2])[0]
