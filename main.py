import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
#
# a = np.array([[1,2],[3,4]])
# #a = np.rot90(a,2)
# g = np.zeros(shape=(4,4,2))
from CifarDataLoader import CifarDataLoader

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
# a = np.zeros(shape= (4,4,30))
# a[0,0,29] = 1
# print(a[3][3][:])
net = ConvolutionalNetwork(10)
net.buildNetwork()
# face = misc.imread('00001.tif')
# x =net.guess(face)
# net.train()
# data_loader = CifarDataLoader()
# images, classes = data_loader.load_training_data()
# print(images.shape)
# names = data_loader.load_class_names()
# x = 13
# print(names[classes[x]])
net.train(5)
net.plot_results()
#
# data = unpickle('cifar-10-batches-py/data_batch_1')
# raw_images = data[b'data']
# classes = np.array(data[b'labels'])
# images = _convert_images(raw_images,3,32)
# plt.imshow(images[x,:,:,:])
# plt.show()

