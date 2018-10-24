# Using claptcha library found on:
# https://github.com/kuszaj/claptcha

import random
import string
from os import listdir
from os.path import isfile, join
from PIL import Image
from claptcha import Claptcha

def randomString():
   rndLetters = (random.choice(string.ascii_uppercase) for _ in range(1))
   return "".join(rndLetters)

onlyfiles = [f for f in listdir("./fonts") if isfile(join("./fonts", f))]

for character in string.ascii_uppercase:
   rnd = random.randint(750, 1000)
   for i in range(rnd):
      c = Claptcha(character, "./fonts/"+onlyfiles[random.randint(0, len(onlyfiles)-1)], (80, 80),
             resample=Image.BICUBIC, noise=0.3, margin=(5,5))
      text, _ = c.write(f'dataset/{c.text}{i:03}.png')
