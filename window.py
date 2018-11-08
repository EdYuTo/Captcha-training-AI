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

background = '#E1E1E1'
global InputInformed
InputInformed = 0


def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()


def main_thread():
    try:
        Console.print_in_prompt('[INFO] Trying to import processed model...')
        model = cPickle.loads(
            open("model.cpickle", "rb").read())  # to read model
        Console.print_in_prompt('[INFO] Success!')
        read = True
        time.sleep(1)
    except:
        Console.print_in_prompt('[INFO] Could not import processed data.')
        read = False
        time.sleep(1)

    if read:
        total = 0
        hit = 0
        Console.print_in_prompt('[INPUT] Enter the number of iterations: ')
        global InputInformed
        while InputInformed == 0:
            time.sleep(1)
        num = InputInformed
        print(num)

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

            ImageShown.update_img()
            Console.print_in_prompt(str('[LOG] I think this image contains: ' +
                                        str(model.predict(rawImages)[0])))

            if str(character) == str(model.predict(rawImages)[0]):
                hit += 1

            time.sleep(0.5)

        os.system('cls' if os.name == 'nt' else 'clear')
        Console.print_in_prompt('[INFO] Tests finished with a precision of: ' +
                                str((hit/total)*100) + "%")
    else:
        Console.print_in_prompt('[INFO] Try to (re)generate model...')


class ImageCaptcha(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        try:
            self.img = ImageTk.PhotoImage(Image.open('test.png'))
            self.panel = Label(self, image=self.img)
            self.panel.pack(fill="both", expand=True)
        except:
            self.img = ImageTk.PhotoImage(Image.open('dataset/X000.png'))
            self.panel = Label(self, image=self.img)
            self.panel.pack(fill="both", expand=True)

    def update_img(self):
        self.img = ImageTk.PhotoImage(Image.open('test.png'))
        self.panel.configure(image=self.img)


class ConsoleIO(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.text = Text(self, wrap="word", height=20, background=background)
        self.text.pack(side=LEFT, fill="both", expand=True)

        self.text.bind("<Return>", self.process_input)
        self.prompt = ""

        self.insert_prompt()

    def print_in_prompt(self, string):
        self.text.delete('1.0', "end")
        self.text.insert("end", "%s" % string)
        self.text.see("end")
        self.insert_prompt()

    def insert_prompt(self):
        c = self.text.get("end-2c")
        if c != "\n":
            self.text.insert("end", "\n")
        self.text.insert("end", self.prompt, ("prompt",))

        self.text.mark_set("end-of-prompt", "end-1c")
        self.text.mark_gravity("end-of-prompt", "left")

    def process_input(self, event=None):
        self.text.insert("end", "")
        command = self.text.get("end-of-prompt", "end-1c")
        self.text.see("end")
        self.insert_prompt()
        global InputInformed
        InputInformed = int(command)
        return 'break'


def restart_program():
    global InputInformed
    InputInformed = 0
    thread.start_new(main_thread, ())


def exit():
    window.destroy()


def gen_model_thread():
    Console.print_in_prompt("[INFO] Describing images...")
    imagePaths = list(paths.list_images("dataset"))

    rawImages = []
    labels = []

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        label = imagePath[8]  # rip hardcode

        pixels = image_to_feature_vector(image)

        rawImages.append(pixels)
        labels.append(label)

        if i > 0 and i % 1000 == 0:
            Console.print_in_prompt(
                "[INFO] Processed {}/{}".format(i, len(imagePaths)))

    rawImages = np.array(rawImages)
    labels = np.array(labels)
    Console.print_in_prompt("[INFO] Pixels matrix: {:.2f}MB".format(
        rawImages.nbytes / (1024 * 1000.0)))

    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.01, random_state=42)

    Console.print_in_prompt("[INFO] Evaluating raw pixel accuracy...")
    model = KNeighborsClassifier(n_neighbors=23, n_jobs=-1)
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    Console.print_in_prompt(
        "[INFO] Raw pixel accuracy: {:.2f}%".format(acc * 100))
    time.sleep(1)

    Console.print_in_prompt("[INFO] Writing model to a file...")
    f = open("model.cpickle", "wb")
    f.write(cPickle.dumps(model))
    f.close()
    Console.print_in_prompt("[INFO] All set, just restart the program!")


def gen_model():
    thread.start_new(gen_model_thread, ())


def center(win):
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
    window = Tk()
    window.title('Captcha Training')
    window.geometry("300x150")
    window.configure(background=background)

    menu = Menu(window)
    menu.add_command(label="GenModel", command=gen_model)
    menu.add_command(label="Restart", command=restart_program)
    menu.add_command(label="Exit", command=exit)
    window.config(menu=menu)

    center(window)

    ImageShown = ImageCaptcha(window)
    Console = ConsoleIO(window)
    ImageShown.pack(side=TOP)
    Console.pack(side=BOTTOM, fill="both", expand=True)

    thread.start_new(main_thread, ())
    window.mainloop()
