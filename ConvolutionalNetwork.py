from CifarDataLoader import CifarDataLoader
from ConvolutionalLayer import ConvolutionalLayer
from FlattenLayer import FlattenLayer
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from PoolingLayer import PoolingLayer
from ReluLayer import ReluLayer
from numpy import linalg as LA
import numpy as np
import matplotlib.pylab as plt

class ConvolutionalNetwork:
    def __init__(self,numClasses):
        self.inputLayer = None
        self.outputLayer = None
        self.numOutputs = numClasses
        self.dataLoader = CifarDataLoader()
        self.error = 0
        self.error_plotX = []
        self.error_plotY = []
        self.precisionX = []
        self.precisionY = []

    def buildNetwork(self):
        self.inputLayer = InputLayer()
        convLayer = ConvolutionalLayer(4,30)
        poolLayer = PoolingLayer(2)
        reluLayer = ReluLayer()
        convLayer2 = ConvolutionalLayer(4,30)
        pool2Layer = PoolingLayer(2)
        flattenLayer = FlattenLayer()
        self.outputLayer = OutputLayer(10)
        flattenLayer.connect(self.outputLayer)
        pool2Layer.connect(flattenLayer)
        convLayer2.connect(pool2Layer)
        reluLayer.connect(convLayer2)
        poolLayer.connect(reluLayer)
        convLayer.connect(poolLayer)
        self.inputLayer.connect(convLayer)

    def guess(self,input):
        self.inputLayer.forwardPropagation(input)
        return self.outputLayer.getOutput()

    def backPropagate(self,expected_output):
        self.outputLayer.backPropagation(expected_output)

    def train(self, numEpochs):
        images, classes = self.dataLoader.load_training_data()
        numImages = images.shape[0]
        self.error_plotX = []
        self.error_plotY = []
        self.error = 0
        for i in range(numEpochs):
            for index in range(numImages):
                expected_output = np.zeros(10)
                output = self.guess(images[index,:,:,:])
                expected_output[classes[index]] = 1
                self.backPropagate(expected_output)
                self.error += (np.power(LA.norm(np.subtract(expected_output, output)), 2)/numImages)
            self.error_plotX.append(i)
            self.error_plotY.append(self.error)
            ratio = self.test_data()
            self.precisionX.append(i)
            self.precisionY.append(ratio)

    def test_data(self):
        images, classes = self.dataLoader.load_training_data()
        classes_names = self.dataLoader.load_class_names()
        numImages = images.shape[0]
        asserts = 0
        for index in range(numImages):
            output = self.guess(images[index,:,:,:])
            if (classes_names[np.argmax(output)]== classes[index]):
                asserts +=1
        return float(float(asserts)/float(numImages))

    def plot_results(self):
        plt.figure()
        plt.title("Precision", fontsize=20)
        plt.xlabel('epochs')
        plt.ylabel('ratio')
        plt.plot(self.precisionX, self.precisionY)
        plt.figure()
        plt.title("Error", fontsize=20)
        plt.xlabel('epochs')
        plt.ylabel('error')
        plt.plot(self.error_plotX, self.error_plotY)
        plt.show()