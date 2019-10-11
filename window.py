from tkinter import *
import tkinter as tk
import random
import string
from claptcha import Claptcha
from os import listdir
from os.path import isfile, join
from PIL import ImageTk, Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import re
import time
import _pickle as cPickle
import _thread as thread
import gdown
import tarfile

background = '#E1E1E1' # window background color
global InputInformed # global variable to read input number (not best approach)
InputInformed = 0
minPixelValue = 200
# image filters
#sharpen_kernel = np.array([0,-1,0,-1,5,-1,0,-1,0])
#edge_kernel = np.array([-1,-1,-1,-1,8,-1,-1,-1,-1])
blur_kernel = np.array([1,1,1,1,1,1,1,1,1])/9.0
#edge_detect_kernel = np.array([0, 1, 0, 1, -4, 1, 0, 1, 0])
#edge_enhance_kernel = np.array([0, 0, 0, -1, 1, 0, 0, 0, 0])
edge_emboss_kernel = np.array([-2, -1, 0, -1, 1, 1, 0, 1, 2])


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()

# loads the model and tests it with some auto-generated samples
def main_thread():
    try: # try to import the model
        Console.print_in_prompt('[INFO] Trying to import processed model...')
        model = cPickle.loads(
            open("model.cpickle", "rb").read())  # to read model
        Console.print_in_prompt('[INFO] Success!')
        read = True # the model was loaded
        time.sleep(1) # delay to read the message
    except:
        Console.print_in_prompt('[INFO] Could not import processed data.')
        read = False # there is no model
        time.sleep(1)  # delay to read the message

    if read:
        total = 0 # number of captchas evaluated
        hit = 0 # number of captchas correctly evaluated
        Console.print_in_prompt('[INPUT] Enter the number of iterations: ')
        global InputInformed
        while InputInformed == 0: # wait until input is typed
            time.sleep(1)
        num = InputInformed

        charList = [] # list of characters that can be used to create a captcha
        charList += string.ascii_uppercase # A-Z
        charList += string.digits # 0-9

        for i in range(num): # make this n times (the number of iterations desired)
            total += 1 # increment the number of total evaluated captchas
            #os.system('cls' if os.name == 'nt' else 'clear')
            rawImages = []

            # lists all available font files
            onlyfiles = [f for f in listdir(
                "dataset-generator/fonts") if isfile(join("dataset-generator/fonts", f))]

            # pick a random character from the list to generate a captcha
            character = charList[random.randint(0, len(charList)-1)]
            #print('Generated image with character: ' + str(character))

            # generate and writes the captcha, respectively
            c = Claptcha(character, "dataset-generator/fonts/"+onlyfiles[random.randint(0, len(onlyfiles)-1)], (80, 80),
                         resample=Image.BICUBIC, noise=0.3, margin=(5, 5))
            text, _ = c.write(f'test.png')

            # this is the evaluation part:
            image = cv2.imread('test.png') # loads the captcha
            pixels = image_to_feature_vector(image) # convert the image to an array

            # apply filters
            pixels = np.convolve(pixels, blur_kernel)
            pixels = np.convolve(pixels, edge_emboss_kernel)
            pixels[pixels > minPixelValue] = 255

            rawImages.append(pixels) # append the array
            rawImages = np.array(rawImages) # convert to array

            ImageShown.update_img() # updates the display with the generated captcha

            # predict() function return the label of the group of images
            Console.print_in_prompt(str('[LOG] I think this image contains: ' +
                                        str(model.predict(rawImages)[0])))

            # if the character was correctly evaluated: +1 hit
            if str(character) == str(model.predict(rawImages)[0]):
                hit += 1

            time.sleep(0.5) # read delay

        # when all the iterations finish, we display the percentage of correctly evaluated captchas
        Console.print_in_prompt('[INFO] Tests finished with a precision of: ' +
                                str((hit/total)*100) + "%")
    else:
        # in case the model could not be loaded
        Console.print_in_prompt('[INFO] Try to (re)generate model...')

# class to represent the image of the captcha on the window
class ImageCaptcha(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        try:
            self.img = ImageTk.PhotoImage(Image.open('test.png')) # loads the image
            self.panel = Label(self, image=self.img) # creates its label
            self.panel.pack(fill="both", expand=True) # link the image to its label
        except:
            self.img = ImageTk.PhotoImage(Image.open('X000.png')) # loads a placeholder
            self.panel = Label(self, image=self.img) # creates its label
            self.panel.pack(fill="both", expand=True) # link the image to its label

    def update_img(self): # updates the displayed image
        self.img = ImageTk.PhotoImage(Image.open('test.png'))
        self.panel.configure(image=self.img)

# class to represent the I/O console
class ConsoleIO(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.text = Text(self, wrap="word", height=20, background=background) # creates its text I/O
        self.text.pack(side=LEFT, fill="both", expand=True) # links it

        self.text.bind("<Return>", self.process_input) # binds the process_input function to '\n' input
        self.prompt = ""

        self.insert_prompt() # updates the displayed text

    def print_in_prompt(self, string): # prints a given string to the console
        self.text.delete('1.0', "end") # similar to the clear function
        self.text.insert("end", "%s" % string) # writes the string
        self.text.see("end")
        self.insert_prompt() # updates the displayed text

    def insert_prompt(self):
        # magic
        c = self.text.get("end-2c")
        if c != "\n":
            self.text.insert("end", "\n")
        self.text.insert("end", self.prompt, ("prompt",))

        self.text.mark_set("end-of-prompt", "end-1c")
        self.text.mark_gravity("end-of-prompt", "left")

    def process_input(self, event=None):
        # only called when the user types something and hit enter
        self.text.insert("end", "")
        command = self.text.get("end-of-prompt", "end-1c")
        self.text.see("end")
        self.insert_prompt()
        global InputInformed 
        InputInformed = int(command) # save the number of iterations to this global
        return 'break'

# used to rerun the program
def restart_program():
    global InputInformed
    InputInformed = 0 # never forget to reset this variable
    thread.start_new(main_thread, ())

# simply exit
def exit():
    window.destroy()

# used to create and write the model file
def gen_model_thread():
    Console.print_in_prompt("[INFO] Describing images...")
    # grab the list of images that we'll be describing
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

        # aplly filters
        pixels = np.convolve(pixels, blur_kernel)
        pixels = np.convolve(pixels, edge_emboss_kernel)
        pixels[pixels > minPixelValue] = 255

        # update the raw images and labels matricies, respectively
        rawImages.append(pixels)
        labels.append(label)

        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            Console.print_in_prompt(
                "[INFO] Processed {}/{}".format(i, len(imagePaths)))

    # show some information on the memory consumed by the raw images matrix
    rawImages = np.array(rawImages)
    labels = np.array(labels)
    Console.print_in_prompt("[INFO] Pixels matrix: {:.2f}MB".format(
        rawImages.nbytes / (1024 * 1000.0)))

    # partition the data into training and testing splits, using 99%
    # of the data for training and the remaining 1% for testing
    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.01, random_state=42)

    # train and evaluate a k-NN classifer on the raw pixel intensities
    Console.print_in_prompt("[INFO] Evaluating raw pixel accuracy...")
    model = KNeighborsClassifier(n_neighbors=23, n_jobs=-1)
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    Console.print_in_prompt(
        "[INFO] Raw pixel accuracy: {:.2f}%".format(acc * 100))
    time.sleep(1)

    # writes the model to disk so it can be accessed later
    Console.print_in_prompt("[INFO] Writing model to a file...")
    f = open("model.cpickle", "wb")
    f.write(cPickle.dumps(model))
    f.close()
    Console.print_in_prompt("[INFO] All set, just restart the program!")

# just used to start the gen_model thread
# this allows the window and the thread to run at the same time
# the window would freeze otherwise
def gen_model():
    thread.start_new(gen_model_thread, ())

# centralize the window
def center(win):
    # also magic
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()


if __name__ == '__main__':
    window = Tk() # creates an instance
    window.title('Captcha Training') # names it
    window.geometry("300x150") # gives its dimension
    window.configure(background=background) # gives its color (start of the code)

    menu = Menu(window) # creates its menu
    menu.add_command(label="GenModel", command=gen_model) # creates and binds the genmodel opt/func
    menu.add_command(label="Restart", command=restart_program) # creates and binds the restart opt/func
    menu.add_command(label="Exit", command=exit) # creates and binds the exit opt/func
    window.config(menu=menu)

    center(window) # centralize

    ImageShown = ImageCaptcha(window) # creates the image instance
    Console = ConsoleIO(window) # creates the console instance
    ImageShown.pack(side=TOP) # links it to the window on the upper part
    Console.pack(side=BOTTOM, fill="both", expand=True) # links it to the window on the bottom part

    thread.start_new(main_thread, ()) # starts the task
    window.mainloop() # keeps the window opened
