# insynth

---
![CI/CD](https://github.com/mlxyz/insynth/actions/workflows/ci.yaml/badge.svg)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/069d3759b9e24a468bd4f47c0c3fd02f)](https://www.codacy.com/gh/mlxyz/insynth/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mlxyz/insynth&amp;utm_campaign=Badge_Grade)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=mlxyz_insynth&metric=bugs)](https://sonarcloud.io/summary/new_code?id=mlxyz_insynth)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=mlxyz_insynth&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=mlxyz_insynth)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=mlxyz_insynth&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=mlxyz_insynth)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=mlxyz_insynth&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=mlxyz_insynth)
[![codecov](https://codecov.io/gh/mlxyz/insynth/branch/master/graph/badge.svg?token=UCHS79CXM7)](https://codecov.io/gh/mlxyz/insynth)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fmlxyz%2Finsynth.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fmlxyz%2Finsynth?ref=badge_shield)
[![Documentation Status](https://readthedocs.org/projects/insynth/badge/?version=latest)](https://insynth.readthedocs.io/en/latest/?badge=latest)

# Testing Audio Perturbators
```python
from insynth.perturbators.audio import AudioShortNoisePerturbator
import librosa
from scipy.io.wavfile import write
perturbator = ...
signal, sr=librosa.load(file_name, sr=None)
result = perturbator.apply((signal, sr))
write("output.wav", sr, result)
```

## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fmlxyz%2Finsynth.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fmlxyz%2Finsynth?ref=badge_large)