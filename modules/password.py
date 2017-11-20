import glob
from pathlib import Path
import itertools
import random
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image

import robot_arm
from modules.solver import Solver
from util import *

letter_width = 23
up_buttons = [(letter_width * i + 31, 35) for i in range(5)]
down_buttons = [(letter_width * i + 31, 116) for i in range(5)]
#letter_xs = [(letter_width * i, letter_width * (i + 1)) for i in range(5)]
letter_xs = [letter_width * i for i in range(5)]
submit_button = (down_buttons[2][0], down_buttons[2][1]+20)



def random_color():
    return (random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255))

def match_letter(image, resize = 0, blur = 0, thresh = 0, cont = 0, detail = 0, do_tess = 0):

        #increase accuracy with some blurring and thresholding
        if resize:
            image = cv2.resize(image, (10,10), interpolation=cv2.INTER_AREA)
        if blur:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        if thresh:
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dir = "letters/"
        filename = f"{dir}{len(glob.glob(f'{dir}*.bmp'))}.bmp"
        if cont:
            (im, contours, hierarchy) = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print(contours[1:])
            im=np.ones((im.shape[0],im.shape[1],3), np.uint8) * 255
            for cnt in contours:
                cv2.drawContours(im, [cnt], -1, random_color(), 1)
            image = im
        letter = " "
        if do_tess:
            cv2.imwrite(filename, image)
            letter = pytesseract.image_to_string(Image.open(filename), config="-psm 10 letters")
        #DEBUG#
            #print(letter); display(image)
            os.remove(filename)
        return letter, image

def get_lcd(module_image):
    return module_image[59:92,21:]
def get_letter(lcd_image, pos):
    x1 = letter_xs[pos]
    return lcd_image[7:30, x1:x1 + 20]
def get_letters(lcd_image):
    for i in range(5):
        yield get_letter(lcd_image, i)

def label_as_float(label):
    return float(ord(label)-ord("a"))

def label_from_float(label):
    return chr(int(label)+ord("a"))

def shift_image(image):
    """shift image left so its first black pixel is in column 0"""
    #im1 = image.copy()
    min1 = np.argmin(image, axis=0) > 0
    pixels_removed =np.argmax(min1)
    image = np.lib.pad(image[:,pixels_removed:], ((0,0),(0, pixels_removed)), 'constant', constant_values=255)
    #display(np.vstack((im1, image)))
    return image
def get_features(image, flat = True):
    """returns a (1, 23*20) array"""
    image = cv2.GaussianBlur(255-image, (5, 5), 0)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im = np.zeros((23, 20), dtype=np.uint8)
    (_, contours, hierarchy) = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(im, [cnt], -1, 255, -1)
    #print(f"starts at {contours[0][0]} - len {contours[0].shape}")
    #image = shift_image(im)
    #display(im)
    image = im
    if flat:
        #display(im)
        return np.resize(image, (1, 23*20)).astype(np.float32)
    else:
        return image

def increment_letter(robot, pos, after = 0.5):
    robot.moduleto(*up_buttons[pos])
    robot.click(before=0.05, after=after)

class Password(Solver):
    def __init__(self, reget_features = True):
        self.init_passwords()
        self.images :np.array = np.array([])
        self.labelled = {}
        self.labels = np.array([])
        for path in glob.glob("data/modules/password/*.bmp"):
            key = (Path(path).resolve().stem)
            key = ord(key)-ord("a")
            im = cv2.imread(path, 0)
            splits = im.shape[0] // 23
            #print (len(np.vsplit(im, splits)))
            for im in np.vsplit(im, splits):
                if reget_features:
                    im = get_features(im, False)
                else:
                    im = shift_image(im)
                self.add_image(im, key)

        self.knn :cv2.ml_KNearest = cv2.ml.KNearest_create()

        #why does np.resize(self.images, (-1, 460)) drop the last item?
        #do it manually
        rows = np.prod(self.images.shape)//460
        traindata = np.resize(self.images, (rows, 460)).astype(np.float32)
        test=traindata[-1]
        test = np.resize(test, (23, 20))
        #display(test)
        #display(self.images)
        labels = self.labels.astype(np.float32)
        #print(labels)
        self.knn.train(traindata , cv2.ml.ROW_SAMPLE, labels)

    def init_passwords(self):
        with open("data/modules/password/passwords.txt") as pfile:
            self.passwords = [line.rstrip("\n") for line in pfile.readlines()]
    def add_image(self, image, label):
        idx = len(self.images)
        #self.images.append(image)
        if idx == 0:
            self.images = image.copy()
            self.labels = np.array([label])
        else:
            self.images = np.vstack((self.images, image))
            self.labels = np.vstack((self.labels, label))
        self.labelled[label_from_float(label)] = self.labelled.get(label_from_float(label), []) + [idx, ]
    def query_bomb(self, robot):
        letters = [[], [], [], [], []]
        for letter_pos in range(5):
            for letter_idx in range(6):
                scrn = robot.grab_selected(0)
                scrn = get_lcd(scrn)
                #display(scrn)
                dir = "lcds/"
                filename = f"{dir}{len(glob.glob(f'{dir}*.bmp'))}.bmp"
                cv2.imwrite(filename, scrn)
                print(filename)
                im = get_letter(scrn, letter_pos)
                letter = self.match_letter(im)
                letters[letter_pos] += [letter]
                print(f"Letter at pos {letter_pos}, idx {letter_idx} is {letter}")
                increment_letter(robot, letter_pos, after=.25)
        return letters

    def fake_query_bomb(self):
        filenames = list(glob.glob("lcds/*.bmp"))
        assert len(filenames) == 30
        letters = [[], [], [], [], []]
        for letter_pos in range(5):
            for letter_idx in range(6):
                im = cv2.imread(filenames[letter_pos*6+letter_idx], 0)
                letters[letter_pos] += self.match_letter(get_letter(im, letter_pos))
        return letters

    def get_solution(self, letters):
        words = (itertools.product(*letters))
        for word in words:
            if "".join(word) in self.passwords:
                print(f"Password: {''.join(word)}")
                return word
        else:
            print("No valid password found")
        return ""
    def apply_solution(self, letters, sol, robot):
        for pos, char in enumerate(sol):
                for i in range(letters[pos].index(char)):
                    increment_letter(robot, pos, after=.1)
        robot.moduleto(*submit_button)
        robot.click()
        print("I am the best")
    def solve(self, robot:robot_arm.RobotArm):
        if robot is None:
            letters = self.fake_query_bomb()
        else:
            letters = self.query_bomb(robot)
        solution = self.get_solution(letters)
        if not (robot is None):
            self.apply_solution(letters, solution, robot)
        #print(letters)

    def match_letter(self, image):
        features = get_features(image)
        #(get_features(image, False))
        ret, result, neighbours, dist = self.knn.findNearest(features, k=1)
        #print (f"distance norm: {dist//1000}")
        return label_from_float(ret)

    def identify(self, image, isLCD = False):
        #t = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #grab just the LCD screen
        if not isLCD:
            image = get_lcd(image)
        display(image)
        letters = [[], [], [], [], []]
        for letter_pos in range(0,5):
            #display(scrn)
            im = get_letter(image, letter_pos)
            letter, im = match_letter(im, resize=0, blur=0 ,thresh=1, cont = 0, detail = 5)
            letters[letter_pos] += [letter]
            print(f"Letter at pos {letter_pos} is {letter}")
            im = cv2.resize(im, None, fx=4,fy=4, interpolation=cv2.INTER_NEAREST)
            #display(im)
        print(f"password: {letters}")
        #print(time.time()-t)
        #22,56
        #130, 92

    def learn(self,image):
        for let in get_letters(image):
            _, letter = match_letter(let, thresh=1)
            value = self.knn.findNearest(np.reshape(letter, (1,460)).astype(np.float32), k=1)
            ret, result, neighbours, dist = value
            #print(f"{chr(int(ret)+ord('a'))} dist {dist[0]}")
            #print(ret)



    def request_supervision(self, image):
        for let in get_letters(image):
            scaled = cv2.resize(let, (240, 240), interpolation=cv2.INTER_NEAREST)
            labelf = read_and_display(scaled, "identify")-ord("a")
            if labelf == ord(" ")-ord("a"):
                raise KeyboardInterrupt("User got bored")
            label = label_from_float(labelf)
            predicted = label_from_float(self.knn.findNearest(get_features(let), k=1)[0])
            _, let = cv2.threshold(let, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if label == predicted:
                print(f"good, I thought it was {label} too")
                continue
            if label in self.labelled:
                print(f"Adding to existing tags for {label}")
                out = cv2.imread(f"data/modules/password/{label}.bmp", 0)
                let = np.vstack((out,let))
                cv2.imwrite(f"data/modules/password/{label}.bmp", let)
                #print(f"But I already have tags for {label}. I'm confused and sad")
            else:
                print(f"I don't have tags for {label}. Creating")
                self.add_image(let, labelf)
                cv2.imwrite(f"data/modules/password/{label}.bmp", let)
    def train(self, image):
        text = ""
        for let in get_letters(image):
            text += self.match_letter(let)
        print(str.upper(text))
        val = read_and_display(image, text=text)
        if val != ord("y"):
            self.request_supervision(image)
        cv2.destroyAllWindows()

        #don't do this
        for k in ():# self.labelled:
            #print(f"{k}: {len(self.labelled[k])}")
            pics = self.labelled[k]
            out = self.images[pics[0]]
            for pic in pics[1:]:
                out = np.concatenate((out, self.images[pic]))
            cv2.imwrite(f"data/modules/password/{k}.bmp", out)