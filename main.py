import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt

a = np.array([[[1,2,3],[3,4,3],[1,2,3]],[[5,6,3],[7,8,3],[1,2,3]]])
#a = np.rot90(a,2)
b = np.array([[1,1],[1,1]])

#x = signal.convolve2d(a,b,'full')

# print(a.shape)
face = misc.imread('00001.tif')
# print(face.shape)
# plt.imshow(face)
# plt.show()
j=np.zeros(shape=(2,2,3))

print(j[:,:,1])