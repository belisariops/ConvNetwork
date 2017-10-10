from NeuralLayer import NeuralLayer
import numpy as np

class PoolingLayer(NeuralLayer):
    def __init__(self,windowSize):
        super().__init__()
        self.windowSize = windowSize
        self.markNeurons = []

    def forwardPropagation(self,input):
        inputHeight,inputWidth,channels = input.shape
        outputHeight = inputHeight/self.windowSize
        outputWidth = inputWidth/self.windowSize
        self.deltas = []
        self.outputFeatureMap = np.zeros(shape=(outputHeight,outputWidth,channels))
        #ojo aca
        for channel in range(channels):
            positionX =0
            positionY = 0
            self.deltas.append(np.zeros(inputHeight,inputWidth))
            for h in range(outputHeight):
                for w in range(outputWidth):
                    for index in range(self.windowSize):
                        window = input[index: index + positionY + 1,index: index + positionX + 1,channel]
                        self.outputFeatureMap[h,w,channel] = np.amax(window)
            positionX += outputHeight
            positionY += outputWidth
        self.nextLayer.forwardPropagation(self.outputFeatureMap)

    def updateWeights(self,input):
        pass

    def backPropagation(self):
        return


