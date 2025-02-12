import random
import numpy

from math import exp

from AbstractNeuron import AbstractNeuron


class SigmoidNeuron(AbstractNeuron):
    def __init__(self):
        super(SigmoidNeuron, self).__init__()
        self.activation_function = lambda z: (1.0 / (1.0 + exp(-z)))
        self.C = 0.01
        self.output = 0
        self.delta = 0
        self.bias = random.uniform(0, 3)

    def getOutput(self, inputs):
        z = 0
        for i in range(len(self.weights)):
            try:
                x = len(self.weights[i]) > 1
            except TypeError:
                x = ""
            z = z + self.weights[i] * inputs[i]
        z = z + self.bias

        self.output = self.activation_function(z)

        return self.output

    def setRandomParameters(self):
        self.setC(0.01)
        self.setBias(random.uniform(1, 3))


