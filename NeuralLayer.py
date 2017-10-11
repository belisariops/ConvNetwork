from abc import ABC, abstractmethod

class NeuralLayer(ABC):
    def __init__(self):
        self.outputFeatureMap = None
        self.kernels = []
        self.deltas = []
        self.inputFeatureMap = []
        self.nextLayer = None
        self.previousLayer = None

    @abstractmethod
    def forwardPropagation(self,input):
        pass

    def updateWeights(self,input):
        pass

    def applyPropagationChanges(self,input):
        self.updateWeights(input)
        self.nextLayer.applyPropagationChanges(self.outputFeatureMap)

    @abstractmethod
    def backPropagation(self):
        pass

    def connect(self,layer):
        self.nextLayer = layer
        layer.previousLayer = self