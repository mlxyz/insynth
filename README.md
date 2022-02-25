<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
![CI/CD](https://github.com/mlxyz/insynth/actions/workflows/ci.yaml/badge.svg)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/069d3759b9e24a468bd4f47c0c3fd02f)](https://www.codacy.com/gh/mlxyz/insynth/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mlxyz/insynth&amp;utm_campaign=Badge_Grade)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=mlxyz_insynth&metric=bugs)](https://sonarcloud.io/summary/new_code?id=mlxyz_insynth)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=mlxyz_insynth&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=mlxyz_insynth)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=mlxyz_insynth&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=mlxyz_insynth)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=mlxyz_insynth&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=mlxyz_insynth)
[![codecov](https://codecov.io/gh/mlxyz/insynth/branch/master/graph/badge.svg?token=UCHS79CXM7)](https://codecov.io/gh/mlxyz/insynth)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fmlxyz%2Finsynth.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fmlxyz%2Finsynth?ref=badge_shield)
[![Documentation Status](https://readthedocs.org/projects/insynth/badge/?version=latest)](https://insynth.readthedocs.io/en/latest/?badge=latest)




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/mlxyz/insynth">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">InSynth</h3>

  <p align="center">
    Robustness testing of Keras models using domain-specific input generation in Python
    <br />
    <a href="https://insynth.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/mlxyz/insynth">View Demo</a>
    ·
    <a href="https://github.com/mlxyz/insynth/issues">Report Bug</a>
    ·
    <a href="https://github.com/mlxyz/insynth/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

The robustness of machine learning models is crucial to their safe and reliable operation in real-world applications.
However, conducting robustness tests is hard as it requires evaluating the model under test repeatedly on different
datasets.

_InSynth_ provides an easy-to-use, efficient and reliable framework for conducting robustness tests.

It works by applying a set of domain-specific input generation techniques (image, audio or text) to the seed dataset,
and then evaluating the model under test on the generated inputs. Then, a set of coverage criteria are evaluated to
determine how well each dataset covers the model. Finally, a report is generated comparing the models' performance and
coverage on different generated datasets.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

  * [Python](https://www.python.org/)
  * [Numpy](https://numpy.org/)
  * [Keras](https://keras.io/)
  * [Pillow](https://python-pillow.org/)
  * [librosa](https://github.com/librosa/librosa)
  * [audiomentations](https://github.com/iver56/audiomentations)
  * [albumentations](https://albumentations.ai/)
  * [scipy](https://scipy.org/)
  * [pandas](https://pandas.pydata.org/)
  * [pydub](https://github.com/jiaaro/pydub)
  * [tqdm](https://github.com/tqdm/tqdm)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

This section describes the steps to follow when you want to get started with the _InSynth_ project.

### Prerequisites

Before installing _InSynth_, make sure you have the following software applications installed and updated to the latest
version.

  * [Python](https://www.python.org/)
  * [pip](https://pip.pypa.io/en/stable/)
  * [ffmpeg](https://www.ffmpeg.org/)

### Installation

To install _InSynth_, only one step is required.

Run the following command to install the python package from the PyPI repository:

   ```sh
   pip install insynth
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->

## Usage

_InSynth_ can be used in a variety of ways depending on the goal you are trying to achieve.

### Data Generation
To mutate an existing dataset using any of the perturbators provided in the framework, follow the steps below.

1. Import the perturbator (e.g. the `ImageNoisePerturbator`) from the respective module.
      ````python
    from insynth.perturbators.image import ImageNoisePerturbator
      ````
2. Create an instance of the perturbator.
      ````python
    perturbator = ImageNoisePerturbator()
      ````
3. Create a `PIL` image object from a file stored on disk and apply the perturbator to it.
      ````python
      seed_image = Image.open('path/to/image.jpg')
      mutated_image = perturbator.apply(seed_image)
      ````
    For audio perturbators, the same procedure applies but using the `librosa.load` method.
    Similarly, text perturbators expect the seed text to be provided as a string.
4. Save the mutated image to disk or display it.
      ````python
      mutated_image.save('path/to/mutated_image.jpg')
      mutated_image.show()
      ````
   
### Coverage Criteria Calculation
To calculate the coverage criteria for a model, follow the steps below.

1. Import the coverage criteria (e.g. the `CoverageCriteria`) from the respective module.
      ````python
    from insynth.metrics.neuron import StrongNeuronActivationCoverageCalculator
      ````
2. Create an instance of the coverage criteria and pass the model to be tested to the constructor.
      ````python
    coverage_calculator = StrongNeuronActivationCoverageCalculator(model)
      ````
   If applicable, run the `update_neuron_bounds` method to determine the neuron bounds of the model.
      ````python
    coverage_calculator.update_neuron_bounds(training_dataset)
      ````
3. Run the `update_coverage` method to update model coverage for the given input.
      ````python
    coverage_calculator.update_coverage(input_data)
      ````
4. Run the `get_coverage` method to retrieve the current model coverage.
      ````python
    coverage = coverage_calculator.get_coverage()
      ````
5. Print the coverage to the console.
      ````python
    print(coverage)
      ````

### Robustness Testing
The previous two sections describe how to generate a mutated dataset and calculate the coverage criteria for a model.
These are prerequisites for testing the robustness of a model.
In order to conduct a full end-to-end robustness test, the runner class is provided in _InSynth_.

1. Import the runner class from the respective module.
      ````python
    from insynth.runners import BasicImageRunner
      ````
2. Create an instance of the runner class and pass the list of perturbators, the list of coverage calculators and the model to the constructor in addition to the dataset inputs and target variables.
      ````python
    runner = BasicImageRunner(list_of_perturbators, list_of_coverage_calculators, dataset_x, dataset_y, model)
      ````
    Note that the `dataset_x` parameter should be a method returning a python generator iterating over all samples to enable the processing of large datasets which do not fit into memory.
    ````python
   dataset_x = lambda: (x for x in dataset)
   ````
3. Run the `run` method to conduct the end-to-end robustness test.
      ````python
    report, robustness_score = runner.run()
      ````
4. Use the `report` variable to analyse the test results or use the `robustness_score` variable to retrieve a single robustness measure of the model.
      ````python
    print(report)
    print(robustness_score)
      ````
If you want to apply all available perturbators and coverage calculators for a given domain, utilize the respective `ComprehensiveRunner` classes.

For more examples, please refer to the [Documentation](https://insynth.readthedocs.io/en/latest/)

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also
simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

  1. Fork the Project
  2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
  3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
  4. Push to the Branch (`git push origin feature/AmazingFeature`)
  5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fmlxyz%2Finsynth.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fmlxyz%2Finsynth?ref=badge_large)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->

## Contact

Marian Lambert

Project Link: [https://github.com/mlxyz/insynth](https://github.com/mlxyz/insynth)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/mlxyz/insynth.svg?style=for-the-badge

[contributors-url]: https://github.com/mlxyz/insynth/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/mlxyz/insynth.svg?style=for-the-badge

[forks-url]: https://github.com/mlxyz/insynth/network/members

[stars-shield]: https://img.shields.io/github/stars/mlxyz/insynth.svg?style=for-the-badge

[stars-url]: https://github.com/mlxyz/insynth/stargazers

[issues-shield]: https://img.shields.io/github/issues/mlxyz/insynth.svg?style=for-the-badge

[issues-url]: https://github.com/mlxyz/insynth/issues

[license-shield]: https://img.shields.io/github/license/mlxyz/insynth.svg?style=for-the-badge

[license-url]: https://github.com/mlxyz/insynth/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/linkedin_username

[product-screenshot]: images/screenshot.png
