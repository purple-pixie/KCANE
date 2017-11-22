from enum import Enum
import numpy as np
import robot_arm
from modules.solver import Solver, SolverModule
from util import *
from math import pi
from matplotlib import pyplot as plt

import logging
log = logging.getLogger(__name__)

#==========================#
#Useful numbers#
a, b, c = (1, 2, 4)
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

class Sequence(SolverModule):
    def __init__(self):
        pass
    def new(self, robot:robot_arm.RobotArm):
        return SequenceSolver(robot)
    def identify(self, image):
        return False



class SequenceSolver(Solver):
    def __init__(self, robot:robot_arm.RobotArm):
        self.robot = robot
        self.wires = {"red": 0, "blue": 0, "black": 0}

    def test_wire(self, color, terminal:TERM):
        self.wires[color] += 1
        return terminal.value & cuts[color][self.wires[color]]

    def cut_wire(self, pos):
        self.robot.moduleto(*wire_starts[pos])
        self.robot.click(after=0.1)
    def next_panel(self):
        self.robot.moduleto(*down_button)
        self.robot.click(after=2)

    def solve(self):
        for panel in range(4):
            for wire in self.get_cuts():
                log.info(f"cutting {wire+1}")
                self.cut_wire(wire)
            log.info(f"next panel")
            self.next_panel()

    def get_cuts(self):
        self.update_image()
        image = self.get_wire_region()
        #image = cv2.imread("fail/img1.bmp")
        #display(image)
        panel = {}
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
                            dump_image(image, "fail/")
                    panel[start] = (color, term)
                    log.debug(f"({x1}, {y1}, {x2}, {y2}): {start} -> {term} ({color})")
        for start in sorted(panel.keys()):
            color, term = panel[start]
            cut = self.test_wire(color, term)
            log.debug(f"{color} wire from {start+1} to {str.upper(term.name)}: {'Make Cut' if cut else 'Don''t cut'}")
            if cut:
                yield start


    def get_wire_region(self):
        return self.image[40:110,30:100]

