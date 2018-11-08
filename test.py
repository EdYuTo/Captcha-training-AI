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
   print('[INFO] Trying to import processed data...')
   model = cPickle.loads(open("model.cpickle", "rb").read())  # to read model
   print('[INFO] Success!')
   read = True
except:
   print('[INFO] Could not import processed data.')
   read = False

if read:
   total = 0
   hit = 0
   num = int(input('[INPUT] Enter the number of iterations: '))
   os.system('cls' if os.name == 'nt' else 'clear')

   charList = []
   charList += string.ascii_uppercase
   charList += string.digits

   for i in range(num):
      total += 1
      phrase = ""
      #os.system('cls' if os.name == 'nt' else 'clear')

      onlyfiles = [f for f in listdir(
          "dataset-generator/fonts") if isfile(join("dataset-generator/fonts", f))]
      
      for i in range(random.randint(2, 4)):
         character = charList[random.randint(0, len(charList)-1)]
         phrase += str(character)

      c = Claptcha(phrase, "dataset-generator/fonts/"+onlyfiles[random.randint(0, len(onlyfiles)-1)], (80*len(phrase), 80),
            resample=Image.BICUBIC, noise=0.3, margin=(5, 5))
      text, _ = c.write(f'test.png')

      img = cv2.imread('test.png')
      height, width = img.shape[:2]
      blocks = int(width/80)
      result = ""
      for i in range(1, blocks+1):
         rawImages = []
         crop_img = img[0:80, 80*(i-1):80*i]
         pixels = image_to_feature_vector(crop_img)
         rawImages.append(pixels)
         rawImages = np.array(rawImages)
         result += str(model.predict(rawImages)[0])

      width = os.get_terminal_size().columns
      print(('[LOG] I think this image contains: ' +
                result).center(width), end="\r")
      
      if phrase == result:
         hit += 1

      time.sleep(2)
   
   os.system('cls' if os.name == 'nt' else 'clear')
   print('[INFO] Tests finished with a precision of: ' +
         str((hit/total)*100) + "%")
else:
   print('[INFO] Run knn.py to generate data...')
