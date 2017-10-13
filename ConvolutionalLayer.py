import numpy as np
from scipy import signal

from NeuralLayer import NeuralLayer


class ConvolutionalLayer(NeuralLayer):
    def __init__(self,kernelSize,numKernels):
        super().__init__()
        for i in range(numKernels):
            x =np.random.random((kernelSize,kernelSize))*2 -1
            self.kernels.append(x)


    def forwardPropagation(self,inputs):
        #Revisar caso en que la imagen es de un canal (blanco y negro)
        try:
            channels = inputs.shape[2]
        except IndexError:
            channels =1
        inputHeigt = inputs.shape[0]
        inputWidth = inputs.shape[1]
        (kernelHeigt, kernelWidth) = self.kernels[0].shape
        resultHeight = inputHeigt - kernelHeigt + 1
        resultWidth = inputWidth - kernelWidth + 1
        self.outputFeatureMap = np.zeros(shape=(resultHeight,resultWidth,len(self.kernels)))
        self.deltas =[]
        for index,kernel in enumerate(self.kernels):
            #Se crea el output y los deltas con 0
            self.deltas.append(np.zeros((resultHeight,resultWidth)))
            self.outputFeatureMap[:,:,index] = np.zeros((resultHeight,resultWidth))
            for channel in range(channels):
                self.outputFeatureMap[:,:,index] = np.add(self.outputFeatureMap[:,:,index],(signal.convolve2d(inputs[:,:,channel],np.rot90(kernel,2), 'valid')))
        self.nextLayer.forwardPropagation(self.outputFeatureMap)

    def updateWeights(self,inputs):
        # Revisar caso en que la imagen es de un canal (blanco y negro)
        try:
            channels = inputs.shape[2]
        except IndexError:
            channels = 1
        count = 0
        for index,delta in enumerate(self.deltas):
            for channel in range(channels):
                akernel = np.copy(self.kernels[index])
                a = signal.convolve2d(inputs[:,:,channel],np.rot90(delta,2),'valid')
                if np.array_equal(a, np.zeros(shape=a.shape)):
                    count +=1
                self.kernels[index] = np.add(self.kernels[index],signal.convolve2d(inputs[:,:,channel],np.rot90(delta,2),'valid'))
        # if count == (len(self.deltas)*channels):
        #     print("aaaaaaaaaaaaaaaaaaaaaaa")


    def backPropagation(self):

        # for kIndex,kernel in enumerate(self.kernels):
        #     for dIndex in range(len(self.previousLayer.deltas)):
        #         self.previousLayer.deltas[dIndex] = np.add(self.previousLayer.deltas[dIndex],signal.convolve2d(np.rot90(kernel,2),self.deltas[dIndex],'full'))
        try:
            for index,previous_delta in enumerate(self.previousLayer.deltas):
                self.previousLayer.deltas[index] = np.zeros(previous_delta.shape)
                for kernel_index,kernel in enumerate(self.kernels):
                    error = signal.convolve2d(self.deltas[kernel_index],np.rot90(kernel))
                    error = np.multiply(error,self.sigmoid(self.previousLayer.outputFeatureMap[:,:,index]))
                    self.previousLayer.deltas[index] = np.add(self.previousLayer.deltas[index],error)
        #print(self.previousLayer.deltas[0].shape)
        #Case that previous layer is an InputLayer
        except AttributeError:
            self.previousLayer.backPropagation()
        self.previousLayer.backPropagation()

