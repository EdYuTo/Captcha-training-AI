import numpy as np
import scipy.signal
import cv2
from imutils import paths
import time

def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()


def feature_vector_to_image(image, size=(32, 32, 3)):
    return img.reshape(size)

sharpen_kernel = np.array([0,-1,0,-1,5,-1,0,-1,0])
edge_kernel = np.array([-1,-1,-1,-1,8,-1,-1,-1,-1])
blur_kernel = np.array([1,1,1,1,1,1,1,1,1])/9.0

edge_detect_kernel = np.array([0, 1, 0, 1, -4, 1, 0, 1, 0])
edge_enhance_kernel = np.array([0, 0, 0, -1, 1, 0, 0, 0, 0])
edge_emboss_kernel = np.array([-2, -1, 0, -1, 1, 1, 0, 1, 2])

#img = np.convolve(img, edge_kernel)
#img = np.convolve(img, sharpen_kernel)
#img = np.convolve(img, edge_detect_kernel)
#img = np.convolve(img, edge_enhance_kernel)

imagePaths = list(paths.list_images("dataset"))

for (i, imagePath) in enumerate(imagePaths):
    print(imagePath)
    img = cv2.imread(imagePath)
    img = image_to_feature_vector(img)

    img = np.convolve(img, blur_kernel)
    img = np.convolve(img, edge_emboss_kernel)

    img[img > 100] = 255

    img = img[:3072]
    img = feature_vector_to_image(img)
    cv2.imwrite('testFilter.png', img)
    time.sleep(0.5)
