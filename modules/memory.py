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


button_data = {"left": 20, "width": 18, "step":24, "top": 95, "bot": 128}
def button_centre(pos):
    return (int(button_data["left"] + button_data["step"] * pos + 12), button_data["top"] + 15)
labels_by_length = [0,3,1,2]
#36, 128
def contour_to_value(cnt):
    length = cv2.arcLength(cnt, False)
    if length < 70:
        return 0
    if length > 120:
        return 2
    if length > 90:
        return 1
    return 3

class Memory(Solver):

    def __init__(self, robot: robot_arm.RobotArm):
        self.robot = robot
        self.update_image()
        self.labels = [-1] * 4
        #moves is a list of (pos, label) pairs
        self.moves = [()] * 5
    def identify(self, image):
        return False
    def do_move(self, pos, stage):
        """click the button at pos pos and remember it as move stage"""
        move = (pos, self.button_label_by_pos(pos))
        print(f"pressing button at {pos+1}, labelled {move[1]+1}")
        self.moves[stage] = move
        self.robot.moduleto(*button_centre(pos))
        self.robot.click(after=4)


    def update_image(self):
        self.image = self.robot.grab_selected(0)
    def get_move_for_stage(self, stage, disp):
        print(f"stage is {stage+1}, disp shows {disp+1}")
        if stage == 0:
            if disp < 2:
                return 1
            return disp
        if stage == 1:
            if disp == 0:
                return self.button_pos_by_label(3)
            if disp == 2:
                return 0
            return self.get_previous_pos(0)
        if stage == 2:
            if disp == 2:
                return 2
            if disp == 3:
                return self.button_pos_by_label(3)
            print(f"this is wrong somehow - disp is {disp+1}, 1- = {2-disp} and moves is {self.moves}")
            return self.get_by_previous_label(1-disp)
        if stage == 3:
            if disp == 0:
                return self.get_previous_pos(0)
            if disp == 1:
                return 0
            return self.get_previous_pos(1)
        if stage == 4:
            if disp == 3:
                return self.get_by_previous_label(2)
            if disp == 2:
                return self.get_previous_label(3)
            return self.get_by_previous_label(disp)


    def solve(self):
        for stage in range(5):
            self.populate_button_labels()
            disp = self.get_display_value()
            try:
                self.do_move(self.get_move_for_stage(stage, disp), stage=stage)
            except:
                print("that did not work")
                self.do_move(self.get_move_for_stage(stage, disp), stage=stage)


    def get_by_previous_label(self, stage):
        """get the pos of the button with the same label as the one pressed on move stage"""
        return self.button_pos_by_label(self.moves[stage][1])
    def get_previous_pos(self, stage):
        """get the pos pressed on move stage"""
        return self.moves[stage][0]
    def button_pos_by_label(self, label):
        """get a button's pos by its label"""
        return self.labels.index(label)
    def button_label_by_pos(self, pos):
        """get the label of the button at pos"""
        return self.labels[pos]
    def test(self):
        #cv2.imwrite("mem/test.bmp", self.image)
        self.populate_button_labels()
        return self.get_display_value()
        #self.robot.screen.save_screen("mem/")
        #display(self.image)


    def get_display_value(self):
        region = self.get_display_region()
        _, image = cv2.threshold(region, 200, 255, cv2.THRESH_BINARY)
        cnt = contour(image)[0]
        return contour_to_value(cnt)
    def get_display_region(self):
        """get the lcd screen region"""
        return self.image[28:28+36, 42:42+46]

    def populate_button_labels(self):
        self.image = self.robot.grab_selected(0)
        labels = sorted(self.get_label_lengths())
        for i, label in enumerate(labels):
           # print(f"{i, label}")
            self.labels[label[1]] = labels_by_length[i]
            #print(f"by length: {labels_by_length[i]}, by value: {label[2]}")
        #print(f"labels: {self.labels}")

    def get_label_lengths(self):
        for pos in range(4):
            image = otsu(self.get_button_region(pos))
            cnt = contour(image)[1]
            yield (cv2.arcLength(cnt, False), pos, contour_to_value(cnt))
        ###
        #
       #lenth: 76.21320307254791   4
       #lenth: 109.84061968326569   3
       #lenth: 67.72792172431946    1
       #lenth: 103.84061968326569   2

    #lenth: 115.32590091228485        3
#           lenth: 101.5979791879654  2
#           lenth: 68.55634880065918  1
#           lenth: 73.79898953437805  4

        #print(f"lenth: {cv2.arcLength(cnt, False)}")
        #cv2.drawContours(image, [cnt], -1, random_color(), 1)
        #label = tess(image, remove=False)
        #print(label)
        #display(image, scale=4)


    def get_button_region(self, pos):
        return self.image[button_data["top"]:button_data["bot"],
               button_data["step"] * pos + button_data["left"] :
                button_data["step"] * pos  + button_data["left"] + button_data["width"]]
