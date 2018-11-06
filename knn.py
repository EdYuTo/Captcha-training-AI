from scipy.ndimage import convolve
from imageio import imread
from os import listdir
from os.path import isfile, join

from statistics import mean
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
    # calculate number of elements using filter function
    lessthan_zero = len(list(filter(lambda x: x < 0, matrix)))
    morethan_zero = len(list(filter(lambda x: x > 0, matrix)))
    zero = len(list(filter(lambda x: x == 0, matrix)))
    result = []

    # calculate mean, variance and entropy 
    # using built-in and numpy functions
    m = mean(matrix)
    variance = np.var(matrix, dtype=np.float64)
    entropy = list(map(lambda x: x*math.log(abs(x)+1), matrix))
    entropy = -sum(entropy)

    # append and return array
    result.append(morethan_zero)
    result.append(zero)
    result.append(lessthan_zero)
    result.append(m)
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
        # get image as an array
        mat = np.ravel(imread(img_path+f))

        # for every filter on the list
        for filt in filters:
            # apply convolution and get data
            conv_mat = np.convolve(mat, filt)
            result_array = get_matrix_data(conv_mat)
            # write processed file
            write_file(fp, result_array)

if __name__ == '__main__':
    train_algorithm()
