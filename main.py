import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
#
# a = np.array([[1,2],[3,4]])
# #a = np.rot90(a,2)
# g = np.zeros(shape=(4,4,2))
# g = np.array([[[0,0],[-1,3]],[[2,1],[2,-2]],[[-1,2],[-3,4]]])
# # g[0][0] = [1,2]
# b = np.array([[1,1],[1,1]])
# print(np.reshape(g.flatten(),(3,2,2)))
# print(g)
# i,j = np.unravel_index(a.argmax(), a.shape)
# x = np.multiply(a,b)
# # print(a.shape)
# face = misc.imread('a.png')
# # print(face.shape)
# # plt.imshow(face)
# # plt.show()
# # j=np.zeros(shape=(2,2,3))
from ConvolutionalNetwork import ConvolutionalNetwork
a = np.zeros(shape= (4,4,2))
print(a[3][3][:])
net = ConvolutionalNetwork(10)
net.buildNetwork()
face = misc.imread('00001.tif')
x =net.guess(face)
net.train()
print(x)