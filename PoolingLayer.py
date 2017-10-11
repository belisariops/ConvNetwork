from NeuralLayer import NeuralLayer
import numpy as np

class PoolingLayer(NeuralLayer):
    def __init__(self,windowSize):
        super().__init__()
        self.windowSize = windowSize
        self.markNeurons = []
        self.deltaDirection = []

    def forwardPropagation(self,input):
        # Revisar caso en que la imagen es de un canal (blanco y negro)
        try:
            channels = input.shape[2]
        except IndexError:
            channels = 1
        inputHeight = input.shape[0]
        inputWidth = input.shape[1]
        outputHeight = inputHeight//self.windowSize
        outputWidth = inputWidth//self.windowSize
        self.deltas = []
        self.deltaDirection = []
        self.outputFeatureMap = np.zeros(shape=(outputHeight,outputWidth,channels))
        #ojo aca
        for channel in range(channels):
            positionX =0
            positionY = 0
            self.deltas.append(np.zeros((outputHeight,outputWidth)))
            self.markNeurons.append(np.zeros((inputHeight,inputWidth)))
            self.deltaDirection.append(np.zeros(shape=(outputHeight,outputWidth,2)))
            for h in range(outputHeight):
                for w in range(outputWidth):
                    for index in range(self.windowSize):
                        window = input[index: index + positionY + 1,index: index + positionX + 1,channel]
                        i, j= np.unravel_index(window.argmax(), window.shape)
                        self.deltaDirection[channel][h,w,:] = [i,j]
                        self.markNeurons[channel][i,j] = 1
                        self.outputFeatureMap[h,w,channel] = window[i,j]
            positionX += outputHeight
            positionY += outputWidth
        self.nextLayer.forwardPropagation(self.outputFeatureMap)

    def updateWeights(self,input):
        pass

    def backPropagation(self):
        channels = self.deltas.shape[2]
        for channel in range(channels):
            deltaHeight,deltaWidth  = self.deltas[:,:,channel].shape
            for dh in range(deltaHeight):
                for dw in range(deltaWidth):
                    [h,w] = self.deltaDirection[channel][dh,dw,:]
                    h =int(h)
                    w = int(w)
                    self.previousLayer.deltas[channel][h,w] = self.deltas[dh,dw,channel]
            # self.previousLayer.deltas[channel] = np.multiply(self.previousLayer.deltas[channel],self.markNeurons)

        self.previousLayer.backPropagation()


