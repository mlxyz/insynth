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
from tqdm import tqdm

from insynth.calculators import StrongNeuronActivationCoverageCalculator, \
    KMultiSectionNeuronCoverageCalculator, NeuronCoverageCalculator, NeuronBoundaryCoverageCalculator, \
    TopKNeuronCoverageCalculator, TopKNeuronPatternsCalculator
from insynth.perturbators.audio import AudioBackgroundWhiteNoisePerturbator, AudioPitchPerturbator, \
    AudioClippingPerturbator, AudioVolumePerturbator, AudioEchoPerturbator, \
    AudioShortNoisePerturbator, AudioBackgroundNoisePerturbator, AudioImpulseResponsePerturbator, \
    AudioCompressionPerturbator
from insynth.perturbators.image import ImageNoisePerturbator, ImageBrightnessPerturbator, ImageContrastPerturbator, \
    ImageSharpnessPerturbator, ImageFlipPerturbator, ImageOcclusionPerturbator, ImageCompressionPerturbator, \
    ImagePixelizePerturbator
from insynth.perturbators.text import TextTypoPerturbator, TextCasePerturbator, TextWordRemovalPerturbator, \
    TextStopWordRemovalPerturbator, TextWordSwitchPerturbator, TextCharacterSwitchPerturbator, \
    TextPunctuationErrorPerturbator
from insynth.runners import BasicImageRunner, BasicAudioRunner, BasicTextRunner


class ExtensiveImageRunner(BasicImageRunner):
    def __init__(self, dataset_x, dataset_y, model, snac_data, pre_predict_lambda=None):
        super().__init__(None, None, dataset_x, dataset_y,
                         model, pre_predict_lambda)
        self.snac_data = snac_data
        self.perturbators = self._get_all_perturbators()
        self.coverage_calculators = self._get_all_coverage_calculators(self.model)

    def _get_all_perturbators(self):
        return [ImageNoisePerturbator(p=1.0),
                ImageBrightnessPerturbator(p=1.0),
                ImageContrastPerturbator(p=1.0),
                ImageSharpnessPerturbator(p=1.0),
                ImageFlipPerturbator(p=1.0),
                ImageOcclusionPerturbator(p=1.0),
                ImageCompressionPerturbator(p=1.0),
                ImagePixelizePerturbator(p=1.0)
                ]

    def _get_all_coverage_calculators(self, model):
        calcs = [
            NeuronCoverageCalculator(model),
            StrongNeuronActivationCoverageCalculator(model),
            KMultiSectionNeuronCoverageCalculator(model),
            NeuronBoundaryCoverageCalculator(model),
            TopKNeuronCoverageCalculator(model),
            TopKNeuronPatternsCalculator(model)
        ]
        for calc in calcs:
            update_neuron_bounds_op = getattr(calc, "update_neuron_bounds", None)
            if callable(update_neuron_bounds_op):
                for sample in tqdm(self.snac_data(), desc='Processing SNAC...'):
                    calc.update_neuron_bounds(self.pre_predict_lambda(sample))
        return calcs


class ExtensiveAudioRunner(BasicAudioRunner):
    def __init__(self, dataset_x, dataset_y, model, snac_data, pre_predict_lambda=None):
        super().__init__(None, None, dataset_x, dataset_y,
                         model, pre_predict_lambda)
        self.snac_data = snac_data
        self.perturbators = self._get_all_perturbators()
        self.coverage_calculators = self._get_all_coverage_calculators(self.model)

    def _get_all_perturbators(self):
        return [AudioBackgroundWhiteNoisePerturbator(p=1.0),
                AudioCompressionPerturbator(p=1.0),
                AudioPitchPerturbator(p=1.0),
                AudioClippingPerturbator(p=1.0),
                AudioVolumePerturbator(p=1.0),
                AudioEchoPerturbator(p=1.0),
                AudioShortNoisePerturbator(p=1.0),
                AudioBackgroundNoisePerturbator(p=1.0),
                AudioImpulseResponsePerturbator(p=1.0)
                ]

    def _get_all_coverage_calculators(self, model):
        calcs = [
            NeuronCoverageCalculator(model),
            StrongNeuronActivationCoverageCalculator(model),
            KMultiSectionNeuronCoverageCalculator(model),
            NeuronBoundaryCoverageCalculator(model),
            TopKNeuronCoverageCalculator(model),
            TopKNeuronPatternsCalculator(model)
        ]
        for calc in calcs:
            update_neuron_bounds_op = getattr(calc, "update_neuron_bounds", None)
            if callable(update_neuron_bounds_op):
                for sample in tqdm(self.snac_data(), desc='Processing SNAC...'):
                    calc.update_neuron_bounds(self.pre_predict_lambda(sample))
        return calcs


class ExtensiveTextRunner(BasicTextRunner):
    def __init__(self, dataset_x, dataset_y, model, snac_data, pre_predict_lambda=None):
        super().__init__(None, None, dataset_x, dataset_y,
                         model, pre_predict_lambda)
        self.snac_data = snac_data
        self.perturbators = self._get_all_perturbators()
        self.coverage_calculators = self._get_all_coverage_calculators(self.model)

    def _get_all_perturbators(self):
        return [TextTypoPerturbator(p=1.0),
                TextCasePerturbator(p=1.0),
                TextWordRemovalPerturbator(p=1.0),
                TextStopWordRemovalPerturbator(p=1.0),
                TextWordSwitchPerturbator(p=1.0),
                TextCharacterSwitchPerturbator(p=1.0),
                TextPunctuationErrorPerturbator(p=1.0),
                ]

    def _get_all_coverage_calculators(self, model):
        calcs = [
            NeuronCoverageCalculator(model),
            StrongNeuronActivationCoverageCalculator(model),
            KMultiSectionNeuronCoverageCalculator(model),
            NeuronBoundaryCoverageCalculator(model),
            TopKNeuronCoverageCalculator(model),
            TopKNeuronPatternsCalculator(model)]
        for calc in calcs:
            update_neuron_bounds_op = getattr(calc, "update_neuron_bounds", None)
            if callable(update_neuron_bounds_op):
                for sample in tqdm(self.snac_data(), desc='Processing SNAC...'):
                    calc.update_neuron_bounds(self.pre_predict_lambda(sample))
        return calcs
