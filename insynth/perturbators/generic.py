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

import warnings

import numpy as np
import scipy.stats as st

from insynth.perturbators.abstract_perturbator import AbstractBlackboxPerturbator


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
        """
        Fit the perturbator to the given dataset.
        :param dataset:
        :return:
        """
        self.distributions = [self.best_fit_distribution(column) for column in dataset.transpose()]

    def best_fit_distribution(self, data, bins=200):
        """
        Returns the best fit distribution to the data.
        :param data:
        :param bins:
        :return:
        """

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
