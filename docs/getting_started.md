# Getting Started

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
This will install the latest version of _InSynth_ to your local Python environment.

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
