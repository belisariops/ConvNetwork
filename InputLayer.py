from NeuralLayer import NeuralLayer


class InputLayer(NeuralLayer):
    def __init__(self):
        # self.inputShape = inputShape
        self.nextLayer = None
        self.input = None

    def updateWeights(self,input):
        pass

    def forwardPropagation(self,input):
        self.input = input
        self.nextLayer.forwardPropagation(input)

    def backPropagation(self):
        self.applyPropagationChanges(self.input)