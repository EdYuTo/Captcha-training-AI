import random
import string
from os import listdir
from os.path import isfile, join
from PIL import Image
from claptcha import Claptcha

onlyfiles = [f for f in listdir("dataset-generator/fonts")
             if isfile(join("dataset-generator/fonts", f))]

character = input('Type character : ')

c = Claptcha(character, "dataset-generator/fonts/"+onlyfiles[random.randint(0, len(onlyfiles)-1)], (80, 80),
             resample=Image.BICUBIC, noise=0.3, margin=(5, 5))
text, _ = c.write(f'./test.png')
