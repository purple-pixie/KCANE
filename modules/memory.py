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
from util import *
import logging
log = logging.getLogger(__name__)


#======================#
#Useful Numbers#

#button positions - step is the x-step from one button to the next, width is the interesting width of a button
#(we don't want to capture the whole of x-step because it will leave ugly vertical lines which break identification)
button_data = {"left": 20, "width": 18, "step":24, "top": 95, "bot": 128}

#looks like a helper function but really it's numbers
def button_centre(pos):
    return (int(button_data["left"] + button_data["step"] * pos + 12), button_data["top"] + 15)

#the labels sorted by arcLength (ascending)
#note that label 0 means "1"  etc. because Python zero-indexing vs the game's 1-indexing
labels_by_length = [0,3,1,2]



#======================#
#helper functions#
def contour_to_value(cnt):
    length = cv2.arcLength(cnt, False)
    if length < 70:
        return 0
    if length > 120:
        return 2
    if length > 90:
        return 1
    return 3


class Solver():
    def __init__(self):
        #nothing to init, no comparison images et c.
        #don't really *need* a Memory/MemorySolver divide, but we want to keep it modular
        pass
    def new(self, robot:robot_arm.RobotArm):
        return MemorySolver(robot)
    def identify(self, robot):
        im = robot.grab_selected().copy()
        hsv = to_hsv(im)
        mask = inRangePairs(hsv, [(17, 24), (40, 125), (134, 255)])
        cnts = contour(mask,  draw=False, mode=cv2.RETR_EXTERNAL)
        rects = list(rectangles(cnts, lambda x,y,w,h: 18<w<28 and 35<h<45))
        for x,y,w,h in rects:
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,0,255), 1)
        return len(rects) == 4, im

class MemorySolver():
    def __init__(self, robot:robot_arm.RobotArm):
        self.robot = robot
        self.update_image()
        self.labels = [-1] * 4
        #moves is a list of (pos, label) pairs
        self.moves = [()] * 5
    def do_move(self, pos, stage):
        """click the button at pos pos and remember it as move stage"""
        move = (pos, self.button_label_by_pos(pos))
        #LOG#print(f"pressing button at {pos+1}, labelled {move[1]+1}")
        self.moves[stage] = move
        self.robot.moduleto(*button_centre(pos))
        self.robot.click(after=4 if stage < 4 else 0)

    def update_image(self):
        self.image = self.robot.grab_selected(0)

    def get_move_for_stage(self, stage, disp):
        """Get the correct button position to press given stage and the number on the display (disp)"""
        log.debug(f"stage is {stage+1}, disp shows {disp+1}")
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
                return self.get_by_previous_label(3)
            return self.get_by_previous_label(disp)


    def solve(self):
        for stage in range(5):
            self.populate_button_labels()
            disp = self.get_display_value()
            move = self.get_move_for_stage(stage, disp)
            log.debug(f"pressing button {move+1} (labelled {self.labels[move]})")
            self.do_move(move, stage=stage)
        return True

    #functions for getting buttons from instructions
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

    def get_display_value(self):
        """get the value displayed on the display"""
        region = self.get_display_region()
        _, image = cv2.threshold(region, 200, 255, cv2.THRESH_BINARY)
        cnt = contour(image,mode=cv2.RETR_EXTERNAL)[0]
        cv2.drawContours(region, [cnt],0,(0,0,0),0)
        #display(region)
        return contour_to_value(cnt)

    def get_button_region(self, pos):
        """get an image of the button at position pos"""
        return self.image[button_data["top"]:button_data["bot"],
               button_data["step"] * pos + button_data["left"] :
                button_data["step"] * pos  + button_data["left"] + button_data["width"]]

    def get_display_region(self):
        """get the display screen region"""
        return self.image[28:28+36, 42:42+46]

    def populate_button_labels(self):
        """identify all the buttons"""
        self.image = self.robot.grab_selected(0)

        #get a list of length, position pairs and sort them by length
        labels = sorted(self.get_label_lengths())
        for i, label in enumerate(labels):
            #this is the Nth button by length, so retrieve its label from the list of labels in length order
            self.labels[label[1]] = labels_by_length[i]
        log.debug(f"labels: {self.labels}")

    def get_label_lengths(self):
        """get the arc lengths of contours on the 4 buttons (for identifying)"""
        for pos in range(4):
            image = 255 - otsu(self.get_button_region(pos))
            #display(image, str(pos), wait_forever=False)
            cnt = contour(image, mode=cv2.RETR_EXTERNAL)[0]
            yield (cv2.arcLength(cnt, False), pos, contour_to_value(cnt))

