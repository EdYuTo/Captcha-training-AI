from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import re
import _pickle as cPickle
import gdown
import tarfile

def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset",
                help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=23,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# try to read the model, create it if there isn't already one
try:
    print('[INFO] Trying to import processed data...')
    model = cPickle.loads(open("model.cpickle", "rb").read())  # to read model
    readModel = 1
    print('[INFO] Success!')
except:
    print('[INFO] Could not import processed data, processing...')
    readModel = 0

# if there is a model, we test it:
if readModel:
    #inputPath = 'dataset/'
    inputPath = input('[INFO] Enter the image path:')
    #inputPath = 'test.png'

    rawImages = []

    image = cv2.imread(inputPath)

    pixels = image_to_feature_vector(image)

    rawImages.append(pixels)

    rawImages = np.array(rawImages)

    # predict() function return the label of the group of images
    # that our image best fit into
    print('[INFO] I think this image contains: ' +
          str(model.predict(rawImages)[0]))

else:  # creates a model and writes it

    # grab the list of images that we'll be describing
    print("[INFO] Describing images...")
    imagePaths = list(paths.list_images("dataset"))
    if len(imagePaths) == 0:
        url = "https://drive.google.com/uc?id=1H8uqq4L81pfEyXT1fb0_R_viKcaV5cW1"
        output = "dataset.tgz"

        gdown.download(url, output, quiet=False)

        tar = tarfile.open(output, "r:gz")
        tar.extractall()
        tar.close()

        os.remove(output)

        imagePaths = list(paths.list_images("dataset"))

    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    rawImages = []
    labels = []

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label (assuming that our
        # path as the format: /dataset/{class}{image_num}.jpg
        image = cv2.imread(imagePath)
        label = imagePath[8]  # rip hardcode

        # extract raw pixel intensity "features"
        pixels = image_to_feature_vector(image)

        # update the raw images and labels matricies, respectively
        rawImages.append(pixels)
        labels.append(label)

        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] Processed {}/{}".format(i, len(imagePaths)))

    #print(labels) # test labels

    # show some information on the memory consumed by the raw images matrix
    rawImages = np.array(rawImages)
    labels = np.array(labels)
    print("[INFO] Pixels matrix: {:.2f}MB".format(
        rawImages.nbytes / (1024 * 1000.0)))

    # partition the data into training and testing splits, using 99%
    # of the data for training and the remaining 1% for testing
    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.01, random_state=42)

    # train and evaluate a k-NN classifer on the raw pixel intensities
    print("[INFO] Evaluating raw pixel accuracy...")
    model = KNeighborsClassifier(
        n_neighbors=args["neighbors"], n_jobs=args["jobs"])
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    print("[INFO] Raw pixel accuracy: {:.2f}%".format(acc * 100))

    # writes the model to disk so it can be accessed later
    f = open("model.cpickle", "wb")
    f.write(cPickle.dumps(model))
    f.close()