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
from modules.solver import Solver, SolverModule
from util import *

#======================#
#Useful Numbers#

letter_width = 23
up_buttons = [(letter_width * i + 31, 35) for i in range(5)]
down_buttons = [(letter_width * i + 31, 116) for i in range(5)]
#letter_xs = [(letter_width * i, letter_width * (i + 1)) for i in range(5)]
letter_xs = [letter_width * i for i in range(5)]
submit_button = (down_buttons[2][0], down_buttons[2][1]+20)


#======================#
#Helper functions#
def pixel_centre(x, y):
    return (int(3 + np.floor(x * 3.5)), int(3 + np.floor(y * 3.5)))
def get_lcd(module_image):
    return module_image[59:92,21:]
def get_letter_region(lcd_image, pos):
    x1 = letter_xs[pos]
    return lcd_image[7:30, x1:x1 + 20]
def get_letter_regions(lcd_image):
    for i in range(5):
        yield get_letter_region(lcd_image, i)

#convert between character identifiers and floating-point values (for the KNN)
def label_as_float(label):
    return (ord(label)-ord("a"))

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
    """convert an image into a vector of its features
    if flat, return it flattened"""
    im = np.zeros((5,5),np.uint8)
    _, a = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for y in range(5):
        for x in range(5):
            xx, yy = pixel_centre(x, y)
            if a[yy][xx] > 120:
                im[y, x] = 255

    #display(cv2.resize(im, (100, 100), interpolation=cv2.INTER_NEAREST))
    return im.ravel('F').astype(np.float32)

class Password(SolverModule):
    def __init__(self, reget_features = True):
        """if reget_features, run get_features on the template images"""
        self.passwords = self.init_passwords()
        self.traindata: np.array = np.array([])
        self.labelled = {}
        self.labels = np.array([], dtype=np.float32)
        #log#print("Reading password training data")
        for path in glob.glob("data/modules/password/*.bmp"):
            key = naked_filename(path)
            if len(key) != 1:
                #log# print(f"{key} is not a valid letter")
                continue
            key = ord(key) - ord("a")
            im = cv2.imread(path, 0)
            splits = im.shape[0] // 23
            # print (len(np.vsplit(im, splits)))
            for im in np.vsplit(im, splits):
                if reget_features:
                    im = get_features(im, False)
                else:
                    im = shift_image(im)
                self.add_image(im, key)

        #log#print("Training KNN")
        self.knn = cv2.ml.KNearest_create()
        print(f"hmmm {self.traindata[-1].shape}")
        #print(self.labels.type())
        import time
        time.sleep(1)
        self.knn.train(self.traindata.astype(np.float32), cv2.ml.COL_SAMPLE, self.labels.astype(np.float32))

        ret, result, neighbours, dist = self.knn.findNearest(self.traindata[-1].astype(np.float32), k=1)

    def new(self, robot:robot_arm.RobotArm):
        return PasswordSolver(robot, self.knn, self.passwords)

    def identify(self, image):
        return False

    def add_image(self, image, label):
        idx = len(self.traindata)
        if idx == 0:
            self.traindata = image.copy()
            self.labels = np.array([label], dtype=np.float32)
        else:
            if len(self.traindata.shape) == 1:
                idx = 1
            self.traindata = np.vstack((self.traindata, image))
            self.labels = np.vstack((self.labels, label))
        self.labelled[label_from_float(label)] = self.labelled.get(label_from_float(label), []) + [idx, ]


    def init_passwords(self):
        try:
            with open("data/modules/password/passwords.txt") as pfile:
                return [line.rstrip("\n") for line in pfile.readlines()]
        except:
            with open("h:/programming/kcane/data/modules/password/passwords.txt") as pfile:
                return [line.rstrip("\n") for line in pfile.readlines()]

class PasswordSolver(Solver):
    def __init__(self, robot:robot_arm.RobotArm = None, knn = None, passwords = []):
        self.knn = knn
        self.passwords = passwords
        self.letters = []
        self.image = None
        self.robot = robot
        self.pointers = [0] * 5

    def increment_letter(self, pos, after=0.5):
        self.robot.moduleto(*up_buttons[pos])
        #self.robot.click(before=0.05, after=after)
        self.pointers[pos] = (self.pointers[pos] + 1) % 6
    def decrement_letter(self, pos, after=0.5):
        self.robot.moduleto(*down_buttons[pos])
        #self.robot.click(before=0.05, after=after)
        self.pointers[pos] = (self.pointers[pos] - 1) % 6
    def seek_letter(self, pos, char):
        """cycle letter at position pos until it shows char
        uses knowledge of the letters in that wheel if present, otherwise identifies letters on the fly"""
        #log##print(f"Looking for {char} from idx {self.pointers[pos]}")
        if char in self.letters[pos]:
            target_idx = self.letters[pos].index(char)
            diff = (self.pointers[pos] - target_idx)
            #print(f"Found at idx {target_idx} (diff: {diff%6})")
            if diff % 6 < 3:
             #   print(f"diff: {diff}, reversing {diff}")
                for i in range(diff):
                    self.decrement_letter(pos, after=0.1)
            else:
              #  print(f"diff: {diff}, advancing {-diff%6}")
                for i in range(-diff%6):
                    self.increment_letter(pos, after=0.1)
        else:
            #TODO remove naked while loop - needs to build up a list for this wheel and make sure we aren't stuck
            while not self.get_letter_label(pos) == char:
               # print(f"Looking for {char} (looking at {self.get_letter_label(pos)})")
                self.increment_letter(pos,after=0.125)


    def query_bomb(self):
        self.update_image()
        self.letters = [[self.get_letter_label(i, False)] for i in range(5)]
        soln =self.get_solution()
        if soln is not None:
            return soln
        for letter_pos in range(5):
            for letter_idx in range(6):
                letters_to_find = ""
                for p in self.passwords:
                    c = p[letter_pos]
                    if c not in self.letters[letter_pos]:
                        if c not in letters_to_find:
                            letters_to_find += c
                #log#print(f"to find in col {letter_pos}: {letters_to_find}")
                if len(letters_to_find) == 0:
                    #debug# log no letters left to find this wheel
                    break
                self.increment_letter(letter_pos, after=.2)
                letter = self.get_letter_label(letter_pos)
                self.letters[letter_pos] += [letter]
                #log#debug#print(self.letters)
                soln =self.get_solution()
                if soln is not None:
                    return soln
            self.passwords = [p for p in self.passwords if p[letter_pos] in self.letters[letter_pos]]
            #log#debug#print(self.passwords)
            #log#debug#print(self.letters)
            #print(f"Letter at pos {letter_pos}, idx {letter_idx} is {letter}")
        return None

    def fake_query_bomb(self):
        filenames = list(glob.glob("lcds/*.bmp"))
        assert len(filenames) == 30
        letters = [[], [], [], [], []]
        for letter_pos in range(5):
            for letter_idx in range(6):
                im = cv2.imread(filenames[letter_pos*6+letter_idx], 0)
                letters[letter_pos] += self.match_letter(get_letter_region(im, letter_pos))
        return letters

    def get_solution(self):
        """get a password we can make from the currently known letters, or None"""
        if len(self.passwords) == 1:
            #log#print(f"Only one option: {self.passwords[0]}")
            return self.passwords[0]
        words = (itertools.product(*self.letters))
        for word in words:
            if "".join(word) in self.passwords:
                #log#print(f"Password: {''.join(word)}")
                return word
        return None
    def apply_solution(self, sol):
        for pos, char in enumerate(sol):
            self.seek_letter(pos, char)
        self.robot.moduleto(*submit_button)
        #self.robot.click()
        #log#print("I am the best")
    def solve(self):
        solution = None
        if self.robot is None:
            letters = self.fake_query_bomb()
        else:
            solution = self.query_bomb()
        #solution = self.get_solution(letters)
        if not (self.robot is None):
            if not solution is None:
                self.apply_solution(solution)
        #print(letters)
    def get_letter_label(self, pos, take_screenshot = True):
        """get the label of letter at pos, if take_screenshot then take a new one first"""
        if take_screenshot:
            self.update_image()
        return self.match_letter(get_letter_region(self.image, pos))
    def match_letter(self, image):
        """get the label for the given image of a letter"""
        features = get_features(image)
        print(features)
        ret, result, neighbours, dist = self.knn.findNearest(features, k=1)
        return label_from_float(ret)
    def update_image(self):
        if not self.robot is None:
            self.image = get_lcd(self.robot.grab_selected(0))
            #dump_image(self.image, "lcds/")
            #self.image = cv2.imread(next(self.ims), 0)

    #below is all legacy training code. Likely doesn't even work any more but it might be re-purposed usefully
    #(the Password/PasswordSolver split hadn't happened when this was written,
    #so some things now belong to the parent class
    def request_supervision(self, image):
        for let in get_letter_regions(image):
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
        for let in get_letter_regions(image):
            text += self.match_letter(let)
        print(str.upper(text))
        val = read_and_display(image, text=text)
        if val != ord("y"):
            self.request_supervision(image)
        cv2.destroyAllWindows()
