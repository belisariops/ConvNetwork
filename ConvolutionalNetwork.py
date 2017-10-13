from CifarDataLoader import CifarDataLoader
from ConvolutionalLayer import ConvolutionalLayer
from FlattenLayer import FlattenLayer
from FullyConnectedLayer import FullyConnectedLayer
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
        convLayer = ConvolutionalLayer(5,10)
        poolLayer = PoolingLayer(4)
        reluLayer = ReluLayer()
        convLayer2 = ConvolutionalLayer(4,20)
        pool2Layer = PoolingLayer(2)
        flattenLayer = FlattenLayer()
        reluLayer2 = ReluLayer()
        fullLayer = FullyConnectedLayer(20)
        self.outputLayer = OutputLayer(self.numOutputs)
        fullLayer.connect(self.outputLayer)
        flattenLayer.connect(fullLayer)
        reluLayer2.connect(flattenLayer)
        pool2Layer.connect(reluLayer2)
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
        file = open('results.txt', 'w')
        images, classes = self.dataLoader.load_training_data()
        numImages = images.shape[0]
        training_images, training_classes = self.dataLoader.load_test_data()
        classes_names = self.dataLoader.load_class_names()
        self.error_plotX = []
        self.error_plotY = []
        for i in range(numEpochs):
            self.error = 0
            print("Epoch {0}: training".format(i+1))
            for index in range(2000):
                if index % 100 == 0:
                    print(index)
                if classes[index] != 0 and classes[index] != 3:
                    continue
                expected_output = np.zeros(10)
                output = self.guess(images[index,:,:,:])
                expected_output[classes[index]] = 1
                self.backPropagate(expected_output)
                self.error += (np.power(LA.norm(np.subtract(expected_output, output)), 2)/numImages)
            self.error_plotX.append(i)
            self.error_plotY.append(self.error)
            print("Epoch {0}: test".format(i+1))
            ratio = self.test_data(training_images,training_classes,classes_names)
            self.precisionX.append(i)
            self.precisionY.append(ratio)
            file.write("Epoch {0}:  precision={1}   error={2}".format(i,ratio,self.error))
        file.close()

    def test_data(self,images,classes ,classes_names):
        numImages = images.shape[0]
        asserts = 0
        total = 0
        for index in range(1000):
            if classes[index] != 0 and classes[index] != 3:
                continue
            total +=1
            output = self.guess(images[index,:,:,:])
            if np.argmax(output) == classes[index]:
                asserts += 1
        print(asserts,total)
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
