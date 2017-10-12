from NeuralLayer import NeuralLayer
import numpy as np

class PoolingLayer(NeuralLayer):
    def __init__(self,windowSize):
        super().__init__()
        self.windowSize = windowSize
        self.markNeurons = []
        self.deltaDirection = []
        self.input = None

    def forwardPropagation(self,input):
        # Revisar caso en que la imagen es de un canal (blanco y negro)
        self.input = input
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
        self.kernels = []
        #ojo aca
        for channel in range(channels):
            positionX =0
            positionY = 0
            self.deltas.append(np.zeros((outputHeight,outputWidth)))
            self.markNeurons.append(np.zeros((inputHeight,inputWidth)))
            self.kernels.append(np.zeros((inputHeight,inputWidth)))
            self.deltaDirection.append(np.zeros(shape=(outputHeight,outputWidth,2)))
            for h in range(outputHeight):
                for w in range(outputWidth):
                    for index in range(self.windowSize):
                        window = input[index: index + positionY + 1,index: index + positionX + 1,channel]
                        i, j= np.unravel_index(window.argmax(), window.shape)
                        self.deltaDirection[channel][h,w,:] = [i,j]
                        self.markNeurons[channel][i,j] = 1
                        self.kernels[channel][i,j] = 1
                        self.outputFeatureMap[h,w,channel] = window[i,j]
            positionX += outputHeight
            positionY += outputWidth
        self.nextLayer.forwardPropagation(self.outputFeatureMap)

    def updateWeights(self,input):
        pass

    def backPropagation(self):
        channels = self.outputFeatureMap.shape[2]
        new_deltas = []
        # print(self.deltas[0].shape)
        # print(self.deltaDirection[0].shape)
        for channel in range(channels):
            delta_height, delta_width = self.deltas[channel].shape
            n_delta = np.zeros(shape=self.kernels[channel].shape)
            for dh in range(delta_height):
                for dw in range(delta_width):
                    #print(self.deltaDirection[channel].shape, delta_height,self.outputFeatureMap.shape)
                    [h, w] = self.deltaDirection[channel][dh, dw, :]
                    h = int(h)
                    w = int(w)
                    n_delta[h, w] = self.deltas[channel][dh, dw]
            new_deltas.append(n_delta)
            # self.previousLayer.deltas[channel] = np.multiply(self.previousLayer.deltas[channel],self.markNeurons)
        self.previousLayer.deltas = new_deltas
        self.previousLayer.backPropagation()


