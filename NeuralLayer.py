from abc import ABC, abstractmethod

class NeuralLayer(ABC):
    def __init__(self, channels):
        self.outputFeatureMap = []
        self.kernels = []
        self.deltas = []
        self.inputFeatureMap = []
        self.nextLayer = None
        self.previousLayer = None

    @abstractmethod
    def forwardPropagation(self,input):
        pass

    @abstractmethod
    def updateWeights(self,input):
        pass

    def applyPropagationChanges(self,input):
        self.updateWeights(input)

    @abstractmethod
    def backPropagation(self):
        pass