from scipy import signal, exp
from scipy import misc
from CifarDataLoader import CifarDataLoader
from ConvolutionalNetwork import ConvolutionalNetwork
import numpy as np
import matplotlib.pyplot as plt

def main():
    network = ConvolutionalNetwork(10)
    network.buildNetwork()
    network.train(10)
    network.plot_results()

if __name__ =="__main__":
    main()