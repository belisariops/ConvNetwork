from ConvolutionalLayer import ConvolutionalLayer
from FlattenLayer import FlattenLayer
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from PoolingLayer import PoolingLayer
from ReluLayer import ReluLayer


class ConvolutionalNetwork:
    def __init__(self,numClasses):
        self.inputLayer = None
        self.outputLayer  = None
        self.numOutputs = numClasses

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
        self.outputLayer.getOutput()
