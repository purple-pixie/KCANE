from enum import Enum
import numpy as np
import robot_arm
from util import *
from math import pi
from matplotlib import pyplot as plt
import screen
import itertools
import logging
log = logging.getLogger(__name__)

#white hsv: (9, 0, 140), (35, 39, 255
bgr_ranges =(['blue', (120, 0, 0), (255, 150, 100)],
               ['white', (140, 133, 150), (255, 255, 255)], #white, misses shadow (catches background but thats croppable)
               ['black', (0,0,0), (30,30,30)], # perfect black (catches background but thats croppable)
               ['red', (0,0,150), (50,50,255)], #good red
               ['yellow', (0,100,100), (50,255,255)], # perfect yellow
               )

draw_colors = { 'blue': (230,30,0), 'white': (255,255,255), 'black': (0,0,0), 'red': (10,20,230), 'yellow': (20,230,230)}

DRAW_DEBUG = False

def find_wires(image):
    #crop to just the wires
    base = image[10:-20, 40:-40]
    canvas = base.copy()
    for label, lower , upper in bgr_ranges:
        #log#print(label)
        mask = inRange(base, lower, upper)
       # display(mask)
        cnts = contour(mask, offset=(40,10))
        if DRAW_DEBUG: contour(mask, draw=True)
        for cnt in cnts:
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect
        #    print(length)
            #   white wires won't connect because of shadowy regions, so they might be length ~100
            #   other wires are going to be ~200
            if (w > base.shape[1]/2) and (4 < h < 46):
                log.debug(f'Found {label} wire size {w, h}')
                cv2.rectangle(canvas,(x+w//2-4,y+h//2-4),(x+w//2+4,y+h//2+4),(0,255,0),1)
                yield y, label, x+w//2, y+h//2
            elif (w * h > 30):
                log.debug(f'{label} line of size {w, h} ignored')
    if DRAW_DEBUG: display(canvas)

class Solver():
    def new(self, robot):
        return SimpleWires(robot)

    def identify(self, robot):
        ##TODO: make sure this isn't being overly restrictive
        #seemed to miss a few
        image = robot.grab_selected()
        canvas = image.copy()
        im = to_hsv(image)
        mask = inRangePairs(im, [(94, 114), (7, 55), (42, 88)] )
        cnts = contour(mask, blur_kernel=(5,5))
        rects = list(rectangles(cnts, lambda x,y,w,h: 12 < w < 22 and 10 < h < 18 and 24 < x < 30))
        for x, y, w, h in rects:
            cv2.rectangle(canvas, (x, y), (x+w,y+h), (0,0,255),1)
        count = len(rects)
        if count == 6:
            return 100, canvas
        if count < 3 or count > 6:
            return False, canvas
        return 50, canvas

def last(li, colour):
    log.debug(f'getting last instance of {colour}')
    return len(li) - 1 - li[::-1].index(colour)

class SimpleWires():
    def __init__(self, robot):
        self.robot = robot
        self.image = self.robot.grab_selected()

    def draw(self, cut = None):
        canvas = np.full(self.image.shape , 120, dtype='uint8')
        for wire in self.wires:
            (y, label, a, b) = wire
            x = 20
            x2=canvas.shape[1]-x
            cv2.rectangle(canvas, (x-8,y-8),(x+8,y+8), (0,0,0), 3)
            cv2.rectangle(canvas, (x2-8, y-8), (x2 + 8, y + 8), (0, 0, 0), 3)
            cv2.line(canvas, (x,y), (x2,y), draw_colors[label], 8)
            if cut == wire:
                cv2.line(canvas, ((a+x)//2,y-20), ((a+x2)//2,y+20), (0,0,255), 5)
                cv2.line(canvas, ((a+x)//2,y+20), ((a+x2)//2,y-20), (0,0,255), 5)
        self.robot.draw_module(canvas)
    def solve(self):
       # DRAW_DEBUG = True
        image = self.robot.grab_selected()
        #image = cv2.imread('img0.bmp')
        self.wires = sorted(find_wires(image))
        self.draw()
        cut = self.get_cut()
        self.draw(cut)

       # DRAW_DEBUG = False
        self.robot.moduleto(*cut[2:])
        self.robot.click()
        return True

    def get_cut(self):
        wires = np.array(self.wires)
        colours = list(wires[:,1])
        colour_counts = {n: colours.count(n) for n in draw_colors}
        odd = self.robot.robot.serial_is_odd

        if len(wires) == 3:
            if colour_counts['red'] == 0:
                return self.wires[1]
            elif colours[2] == 'white':
                return self.wires[2]
            elif colour_counts['blue'] > 1:
                return self.wires[last(colours, 'blue')]
            return self.wires[2]
        elif len(wires) == 4:
            if (colour_counts['red'] > 1) and odd():
                return self.wires[last(colours, 'red')]
            elif (colours[-1] == 'yellow') and not colour_counts['red']:
                return self.wires[0]
            elif colour_counts['blue']==1:
                return self.wires[0]
            elif colour_counts['yellow']>1:
                return self.wires[3]
            return self.wires[1]
        elif len(wires) == 5:
            if colours[-1] == 'black' and odd():
                return self.wires[3]
            elif (colour_counts['red'] == 1) and (colour_counts['yellow'] > 1):
                return self.wires[0]
            elif colour_counts['black'] == 0:
                return self.wires[1]
            return self.wires[0]
        else: # len[wires] == 6
            if colour_counts['yellow'] == 0 and odd():
                return self.wires[2]
            elif colour_counts['yellow']==1 and colour_counts['white']>1:
                return self.wires[3]
            elif colour_counts['red'] == 0:
                return self.wires[5]
            return self.wires[3]
