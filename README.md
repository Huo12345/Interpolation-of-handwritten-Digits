# Interpolation of handwritten digits
This repository is the base for experiment in the visualisation of processes in a neuronal network. The goal is to train a fully connected feed forward neuronal network to recognize digits from the MNIST dataset and in a second step reverse the calculations in the neuronal network and to create images of mixed digits. The abstract is available in the README, the rest of the document can be found in the report folder. Please note that the report has been published in German.
## Abstract
This paper takes a crack at visualizing the processes happening inside a feed forward neuronal network. In a fist approach attempts on inverting the classification happening inside the neuronal network have been made. In this way, classification vectors of mixed digits could be turned into pictures by the inverted version of the neuronal network. This approach had multiple issues, ranging from the mathematical properties of non-squared matrices not always having a perfect inverse to rounding errors of floating-point calculation getting amplified by the activation function. In a second approach randomly generated images were approximated to a target vector by reducing the error of the classification of that image. This method produced seemingly random images opposed to what the estimated error of the network would suggest. This result leads to the believe that the neuronal network has not learned a good concept of digits.
## Installation
1. Clone this repo
1. Install libraries
    1. Numpy (https://scipy.org/install.html)
    1. Tensorflow (https://www.tensorflow.org/install/) 
    1. Matplotlib (https://matplotlib.org/users/installing.html) 
1. Change out_dir and work_dir variable to match your operating system
1. Run the script of your interest in the code digit-interploation/ directort
