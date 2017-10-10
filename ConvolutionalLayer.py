import numpy as np
from scipy import signal

from NeuralLayer import NeuralLayer


class ConvolutionalLayer(NeuralLayer):
    def __init__(self,kernelSize,numKernels):
        super().__init__()
        for i in range(numKernels):
            self.kernels.append(np.random.random(kernelSize,kernelSize))


    def forwardPropagation(self,inputs):
        #Revisar caso en que la imagen es de un canal (blanco y negro)
        try:
            channels = inputs.shape[2]
        except IndexError:
            channels =1
        inputHeigt = inputs.shape[0]
        inputWidth = inputs.shape[1]
        (kernelHeigt, kernelWidth) = self.kernel[0].shape
        resultHeight = inputHeigt - kernelHeigt + 1
        resultWidth = inputWidth - kernelWidth + 1
        self.outputFeatureMap = np.zeros(shape=(resultHeight,resultWidth,len(self.kernels)))
        self.deltas =[]
        for index,kernel in enumerate(self.kernels):
            #Se crea el output y los deltas con 0
            self.deltas.append(np.zeros(resultHeight,resultWidth))
            self.outputFeatureMap[index] = np.zeros(resultHeight,resultWidth)
            for channel in channels:
                if channels!= 1:
                    self.outputFeatureMap[:,:,index] = np.add(self.outputFeatureMap[:,:,index],(signal.convolve2d(inputs[:,:,channels],np.rot90(kernel,2), 'valid')))
                else:
                    self.outputFeatureMap[:,:,index] = (signal.convolve2d(inputs,np.rot90(kernel,2), 'valid'))
        self.nextLayer.forwardPropagation(self.outputFeatureMap)

    def updateWeights(self,inputs):
        for index,delta in enumerate(self.deltas):
            for input in inputs:
                self.kernels[index] = np.add(self.kernels[index],signal.convolve2d(input,np.rot90(delta,2),'valid'))


    def backPropagation(self):
        for kIndex,kernel in enumerate(self.kernels):
            for dIndex in range(len(self.previousLayer.deltas)):
                self.previousLayer.deltas[dIndex] = np.add(self.previousLayer.deltas[dIndex],signal.convolve2d(np.rot90(kernel,2),self.deltas[kIndex],'full'))


