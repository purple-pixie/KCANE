from enum import Enum
import numpy as np
import robot_arm
from util import *
from math import pi
from matplotlib import pyplot as plt
import screen
import logging
log = logging.getLogger(__name__)
from operator import itemgetter
#==========================#
#Useful numbers#
a, b, c = (1, 2, 4)
#cuts to make. cuts["red"][1] is what to do with the first red wire et c.
#if destination & cuts[color][index] then make the cut
cuts = {    "red"  : [0,    c,  b,a,a+c,b,a+c,a+b+c,a+b,b],
            "blue" : [0,    b,a+c,b,  a,b,b+c,    c,a+c,a],
            "black": [0,a+b+c,a+c,b,a+c,b,b+c,  a+b,  c,c] }
class TERM(Enum):
    a = a
    b = b
    c = c

def pos_from_y(y):
    if y < 22:
        return 0
    if y > 45:
        return 2
    return 1

def get_ends(x1, y1, x2, y2):
    #maybe just toss this whole thing and do it interactively.
    #it does work but still ...
    #could just mouse over each start and use the red highlighter for it
    angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
    angle_size = abs(angle)
    start = pos_from_y(y1)
    if angle_size < 10:
        # horizontal wire
        return start, TERM(1 << start)
    elif angle_size < 35:
        # +/- 1 position
        # lines are a bit fiddly, so take the center point and see if it's above or below the middle
        # then check if it's an up or down step
        if (y1 + y2) / 2 < 33:
            if angle > 0:
                return 0, TERM.b
            else:
                return 1, TERM.a
        else:
            if angle > 0:
                return 1, TERM.c
            else:
                return 2, TERM.b
    else:
        if y1 < y2:
            return 0, TERM.c
        else:
            return 2, TERM.a

boundaries = [([0, 0, 150], [50, 50, 255], "red"),
              ([150, 0, 0], [255, 100, 100], "blue"),
              ([0, 0, 0], [15, 15, 15], "black")
              ]
down_button = (65, 133)
wire_starts = ((42, 50), (42, 72), (42, 97))

class Solver():
    def __init__(self):
        pass
    def new(self, robot:robot_arm.RobotArm):
        return SequenceSolver(robot)
    def identify(self, robot):

        xoff = 22
        yoff = 32
        base = robot.grab_selected().copy()
        im = base[yoff: 118, xoff:110]
        hsv = to_hsv(im)
        mask = inRangePairs(hsv, [(101, 139), (0, 51), (0, 105)])
        mask = cv2.GaussianBlur(mask, (5,5),0)
        k=np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, k, iterations=1)
        mask = cv2.threshold(mask, 40,255,cv2.THRESH_BINARY)[1]
        #mask = cv2.dilate(mask, k,iterations=1)
        #  display(mask)
        cnts = contour(mask, draw=False, offset=(xoff, yoff), mode=cv2.RETR_EXTERNAL)
        terms = list(circles(cnts,key=lambda x: 4<x[1]<18 and 32<x[0][0]<50))
        #      terms = list(rectangles(cnts)) # , lambda x, y, w, h: 5 < w < 35 and 5 < h < 25 and 25 < x < 42))
        #       for x,y,w,h in terms:
        #            cv2.rectangle(base, (x,y), (x+w,y+h), (0,255,255))
        for x, y, r in terms:
            cv2.circle(base, (x,y), r, (0,255,0))
        out= hstack_pad(base, to_bgr(gray=mask))
        log.debug(f"wire sequence found {len(terms)} terminals")
        if len(terms) == 3:
            test = self.new(robot)
            wires = len(test.get_panel())
            log.debug(f"wire sequence retrest found {wires} wires")
            return 0 < wires < 4, base
        return False, base

class SequenceSolver():
    def __init__(self, robot:robot_arm.RobotArm):
        self.robot = robot
        self.wires = {"red": 0, "blue": 0, "black": 0}

    def solve_interactive(self):
        pass
        #for each start, take a baseline screenshot (mouse out the way)
        #then mouse over it, subtract base from new
        #cv2.findNonZero() on it, boundingRect that
        #derive terminal from rectangle
        #sample color from around start
        #test_wire, cut or dont

            ##33 - 75 | 1->b
            ##73 - 114 | 3->b
            ##62 - 87 | 2 -> b
            ##32 - 100  | 1->c
            ##34 - 86 | 2 - >a
            ##70 - 114 | 3->b

    def test_wire(self, color, terminal:TERM):
        self.wires[color] += 1
        return terminal.value & cuts[color][self.wires[color]]

    def cut_wire(self, pos):
        self.robot.moduleto(*wire_starts[pos])
        self.robot.click(after=0.1)
    def next_panel(self, wait):
        self.robot.moduleto(*down_button)
        self.robot.click(after=2 if wait else 0)

    def solve(self):
        for panel in range(4):
            for wire in self.get_cuts():
                log.info(f"cutting {wire+1}")
                self.cut_wire(wire)
            log.info(f"next panel")
            self.next_panel(panel < 3)
        return True

    def get_panel(self):
        panel = {}
        self.image = self.robot.grab_selected()
        image = self.get_wire_region()
        for lower, upper, color in boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(image, lower, upper)
            lines = cv2.HoughLinesP(mask, 2, pi/90, 50, minLineLength = 30, maxLineGap = 50)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    start, term = get_ends(x1, y1, x2, y2)
                    if panel.get(start, (color, term)) != (color, term):
                            log.warning(f"Found {(color, term)}, already have {panel[start]} for {start} {line[0]}")
                            return {}
                            #dump_image(image, "fail/")
                    panel[start] = (color, term)
                    log.debug(f"({x1}, {y1}, {x2}, {y2}): {start} -> {term} ({color})")
        return panel

    def get_cuts(self):
        #image = cv2.imread("fail/img1.bmp")
        #display(image)
        panel = self.get_panel()
        for start in sorted(panel.keys()):
            color, term = panel[start]
            cut = self.test_wire(color, term)
            log.debug(f"{color} wire from {start+1} to {str.upper(term.name)}: {'Make Cut' if cut else 'Don''t cut'}")
            if cut:
                yield start


    def get_wire_region(self):
        return self.image[40:110,30:100]

