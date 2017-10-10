import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt

a = np.array([[1,2],[3,4]])
#a = np.rot90(a,2)
print(a)
g = np.zeros(shape=(4,4,2))
g = np.array([[[0,0],[-1,3]],[[2,1],[2,-2]],[[-1,2],[-3,4]]])
# g[0][0] = [1,2]
b = np.array([[1,1],[1,1]])
print(len(g.flatten()))
i,j = np.unravel_index(a.argmax(), a.shape)
x = np.multiply(a,b)
# print(a.shape)
face = misc.imread('00001.tif')
# print(face.shape)
# plt.imshow(face)
# plt.show()
j=np.zeros(shape=(2,2,3))