import warnings

import numpy as np

from insynth.perturbation import AbstractBlackboxPerturbator

from scipy.stats._continuous_distns import _distn_names

import scipy.stats as st


class GenericPerturbator(AbstractBlackboxPerturbator):
    def __init__(self, dataset):
        self.dataset = np.array(dataset)
        self.distributions = [self.best_fit_distribution(column) for column in self.dataset.transpose()]

    def apply(self, original_input):
        return [distribution[0].rvs(*distribution[1]) for distribution in self.distributions]

    def best_fit_distribution(self, data, bins=200):
        """Model data by finding best fit distribution to data"""

        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Best holders
        best_distributions = []

        # Estimate distribution parameters from data
        for ii, distribution in enumerate([d for d in _distn_names if d not in ['levy_stable', 'studentized_range']]):

            distribution = getattr(st, distribution)

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data)
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # identify if this distribution is better
                    best_distributions.append((distribution, params, sse))

            except:
                pass

        return sorted(best_distributions, key=lambda x: x[2])[0]
