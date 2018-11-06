import random
import string
from os import listdir
from os.path import isfile, join
from PIL import Image
from claptcha import Claptcha
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
import time


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


try:
   print('trying to import processed data...')
   model = cPickle.loads(open("model.cpickle", "rb").read())  # to read model
   print('success!')
   read = True
except:
   print('could not import processed data.')
   read = False

if read:
   total = 0
   hit = 0
   num = int(input('Enter the number of iterations: '))
   os.system('cls' if os.name == 'nt' else 'clear')

   charList = []
   charList += string.ascii_uppercase
   charList += string.digits

   for i in range(num):
      total += 1
      #os.system('cls' if os.name == 'nt' else 'clear')
      rawImages = []

      onlyfiles = [f for f in listdir(
          "dataset-generator/fonts") if isfile(join("dataset-generator/fonts", f))]

      character = charList[random.randint(0, len(charList)-1)]
      #print('Generated image with character: ' + str(character))

      c = Claptcha(character, "dataset-generator/fonts/"+onlyfiles[random.randint(0, len(onlyfiles)-1)], (80, 80),
                   resample=Image.BICUBIC, noise=0.3, margin=(5, 5))
      text, _ = c.write(f'test.png')

      image = cv2.imread('test.png')
      pixels = image_to_feature_vector(image)
      rawImages.append(pixels)
      rawImages = np.array(rawImages)

      width = os.get_terminal_size().columns
      print(str('I think this image contains: ' +
                str(model.predict(rawImages)[0])).center(width), end="\r")
      
      if str(character) == str(model.predict(rawImages)[0]):
         hit += 1

      time.sleep(0.5)
   
   os.system('cls' if os.name == 'nt' else 'clear')
   print('Tests finished with a precision of: ' + str((hit/total)*100) + "%")
else:
   print('Run knn.py to generate data...')
