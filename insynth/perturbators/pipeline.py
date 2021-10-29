from insynth.perturbation import AbstractBlackboxPerturbator


class PipelinePerturbator(AbstractBlackboxPerturbator):

    def __init__(self, perturbators: [AbstractBlackboxPerturbator]):
        self.perturbators = perturbators

    def apply(self, original_input):
        output = original_input
        for perturbator in self.perturbators:
            output = perturbator.apply(output)
        return output
