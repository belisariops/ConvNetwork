from NeuralLayer import NeuralLayer


class InputLayer(NeuralLayer):
    def __init__(self):
        # self.inputShape = inputShape
        self.outputFeatureMap = None
        self.nextLayer = None
        self.input = None

    def updateWeights(self,input):
        pass

    def forwardPropagation(self,input):
        self.outputFeatureMap = input
        self.nextLayer.forwardPropagation(self.outputFeatureMap)

    def backPropagation(self):
        self.applyPropagationChanges(self.input)