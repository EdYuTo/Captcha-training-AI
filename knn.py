from scipy.ndimage import convolve
from imageio import imread
from os import listdir
from os.path import isfile, join

import math
import numpy as np

img_path = 'dataset/'
files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
filters = []
fp = open("processed.data", "w+")

filters.append(np.ravel(np.array([[1, 1, 1],[0, 1, 0], [0, 1, 0]])))
filters.append(np.ravel(np.array([[0, 1, 0],[0, 1, 0],[1, 1, 1]])))
filters.append(np.ravel(np.array([[1, 0, 1],[1, 0, 1],[1, 1, 1]])))
filters.append(np.ravel(np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]])))
filters.append(np.ravel(np.array([[-1,  0, -1], [0,  4,  0], [-1,  0, -1]])))
filters.append(np.ravel(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])))
filters.append(np.ravel(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])))

def get_matrix_data(matrix):
    # initialize components used
    morethan_zero = 0
    zero = 0
    lessthan_zero = 0
    mean = 0
    variance = 0
    entropy = 0
    result = []

    for element in matrix:
        if(element < 0):
            lessthan_zero += 1
        elif(element == 0):
            zero += 1
        elif(element > 0):
            morethan_zero += 1
        mean += element
        entropy += (element * math.log(abs(element)+1))

    mean = mean/matrix.size
    entropy *= -1

    for element in matrix:
        variance += (element-mean)**2

    variance = variance/matrix.size

    # append results on array
    result.append(morethan_zero)
    result.append(zero)
    result.append(lessthan_zero)
    result.append(mean)
    result.append(variance)
    result.append(entropy)

    return result

def write_file(file_p, array):
    # write processed image as array on file
    file_p.write(str(array))
    file_p.write('\n')

def train_algorithm():
    # for every dataset file
    for f in files:
        # for every filter on the list
        for filt in filters:
            # get image as an array
            mat = np.ravel(imread(img_path+f))
            # apply convolution and get data
            conv_mat = np.convolve(mat, filt)
            result_array = get_matrix_data(conv_mat)
            # write processed file
            write_file(fp, result_array)

if __name__ == '__main__':
    train_algorithm()
