import unittest

from scipy.stats import norm, chi2
from scipy.stats import stats

from insynth.perturbators.generic import GenericPerturbator


class TestImage(unittest.TestCase):
    def test_GenericPerturbator(self):
        perturbator = GenericPerturbator(p=1.0)

        data = norm.rvs(loc=1.0, scale=2.0, size=10)

        perturbator.fit(data)

        output_data = perturbator.apply(data)

        k2, p = stats.normaltest(output_data)
        alpha = 1e-3
        assert p > alpha

        data = chi2.rvs(df=2, size=10)

        perturbator.fit(data)

        output_data = perturbator.apply(data)

        k2, p = stats.chisquare(output_data)
        alpha = 1e-3
        assert p > alpha


if __name__ == '__main__':
    unittest.main()
