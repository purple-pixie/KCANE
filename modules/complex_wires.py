from util import *
import logging
log = logging.getLogger(__name__)
import time
from enum import Enum
from operator import itemgetter
from robot_arm import RobotArm
import itertools
# y = 19
# x = 23 + idx*16




class WIRE(Enum):
    white = 0
    blue = 1
    red = 2
    star = 4
    led = 8

C="cut"
D="dont"
P="parallel"
S="serial"
B="batteries"
w,b,r,s,l = [a.value for a in WIRE]
cuts = { (WIRE.white.value): "C",
        (WIRE.blue.value): "S",
        (WIRE.red.value): "S",
        (WIRE.star.value): "C",
        (WIRE.led.value): "D",
        (WIRE.blue.value + WIRE.red.value): "S",
        (WIRE.blue.value + WIRE.star.value): "D",
        (WIRE.blue.value + WIRE.led.value): "P",
        (WIRE.red.value + WIRE.star.value): "C",
        (WIRE.red.value + WIRE.led.value): "B",
        (WIRE.star.value + WIRE.led.value): "B",
        (WIRE.blue.value + WIRE.red.value + WIRE.star.value): "P",
        (WIRE.blue.value + WIRE.red.value + WIRE.led.value): "S",
        (WIRE.blue.value + WIRE.star.value + WIRE.led.value): "P",
        (WIRE.red.value + WIRE.star.value + WIRE.led.value): "B",
        (WIRE.blue.value + WIRE.red.value + WIRE.star.value + WIRE.led.value): "D"}

### training
#import itertools
#for length in range(2,5):
#    for i in itertools.combinations(("blue","red","star","led"),length):
#        input(f"{i}: ")
## ('white'): C,
#('blue'): S,
#('red'): S,
#('star'): C,
#('led'): D,
#('blue', 'red'): S,
#('blue', 'star'): D,
#('blue', 'led'): P,
#('red', 'star'): C,
#('red', 'led'): B,
#('star', 'led'): B,
#('blue', 'red', 'star'): P,
#('blue', 'red', 'led'): S,
#('blue', 'star', 'led'): P,
#('red', 'star', 'led'): B,
#('blue', 'red', 'star', 'led'): D


wire_ranges = { WIRE.red:[(160, 179), (88, 255), (0, 255)],
                WIRE.blue: [(110, 130), (88, 255), (0, 255)],
                WIRE.white: [(5, 34), (0, 32), (177, 255)]
                }

def make_pretty(wire):
    color = "White"
    if wire < 0:
        color = "No"
    if (wire & 3) == 3:
        color = "Blue-Red"
    elif wire & 1:
        color = "Blue"
    elif wire & 2:
        color = "Red"
    return f"{color} Wire, {'LED' if wire&8 else 'No LED'}, {'Star' if wire&4 else 'No Star'}"



class ComplexWires():
    def __init__(self, robot:RobotArm):
        self.robot = robot
        self.debug_image = robot.grab_selected().copy()
        self.image = to_hsv(self.debug_image)
        self.wires = [0]*6
        self.leds = [0]*6
        self.stars = [0]*6
        self.valid = False
        if not self.get_stars():
            return
        self.get_wires()
        self.valid = True

    def solve(self):
        moves = (list(self.get_cuts()))
        move_dict = {a:[] for a in "BCDPS"}
        for v, k in moves:
            move_dict[k] = move_dict.get(k, []) + [v]
        print(move_dict)
        log.debug(f"skipping D's: {len(move_dict['D'])} skipped")
        #cut the Cs
        self.cut_wires(move_dict["C"])
        bats = self.robot.robot.gubbins["batteries"]
        if bats > 1:
            log.debug(f"{bats} > 1, cutting {len(move_dict['B'])} B's")
            self.cut_wires(move_dict["B"])
        if self.robot.robot.gubbins["parallel"]:
            log.debug(f"Parallel port found, cutting {len(move_dict['P'])} P's")
            self.cut_wires(move_dict["B"])
        #now the interesting one - OCR is fiddly but if we assume we got everything else right
        #then we can actually infer serial state from the module state because it tells us if it is solved or not
        serials = move_dict["S"]
        if len(serials):
            log.debug(f"{len(serials)} S's found, deducing serial")
            if self.robot.indicator_state() == 1:
                if self.robot.robot.serial_is_odd():
                    log.debug(f"Serial is definitely odd, 'S' wires uncut and module solved")
                else:
                    self.robot.robot.serial_digit_error(False)
                    log.info(f"Module says solved but 'S' wires remain, serial must be wrong")
            else:
                if self.robot.robot.serial_is_odd():
                    self.robot.robot.serial_digit_error(False)
                    log.info(f"Module not solved and 'S' wires remain, serial must be wrong")
                else:
                    log.debug(f"Serial is definitely even, 'S' wires uncut and module still not solved")
                log.debug(f"Cutting {serials} 'S'")
                self.cut_wires(serials)


    def cut_wires(self, indexes):
        for wire in indexes:
            log.info(f"Cutting wire {wire+1}")
            self.robot.moduleto(23+16*wire, 29)
            self.robot.click(after=0.05)
        pass

    def get_cuts(self):
        for i, wire in enumerate(self.wires):
            if wire == -1:
                log.debug(f"no wire in position {i}")
                continue
            wire = wire + self.leds[i] + self.stars[i]
            cut = cuts[wire]
            log.debug(f"{make_pretty(wire)}->{cut}")
            yield i, cut

    def get_stars(self):
        hsv = self.image
        #dont need the indicator
        hsv[:23 , 130:] = 0
        mask = blur(inRangePairs(hsv, [(9, 94), (85, 238), (87, 202)]))
        (im, cnts, hierarchy) = cv2.findContours(mask
                                 ,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        x = 6
        if not len(cnts):
            log.debug(f"found no terminals, not complex")
            self.debug_image=to_bgr(gray=mask)
            return False
        for cnt, data in zip(cnts, hierarchy[0]):
            #contours come from right to left, by rightmost pixel,
            #so the box will always come before anything it contains
            color = (0, 0, 255)
            if data[3] == -1:
                x -= 1
                if x < 0:
                    log.debug(f"found over 6 terminals")
                    return False
                color = (0,255,0)
            else:
                if x == 6:
                    log.debug(f"found contained before container")
                    return False
                self.stars[x] = WIRE.star.value
            cv2.drawContours(self.debug_image, [cnt], -1, color,2)
            #draw_label(out, centrelabel=f"{x}")
        if x > 0:
            log.debug(f"only found {6-x} terminals")
            return False
        log.debug(f"stars: {self.stars}")
        return True

    def get_wires(self):
        image = self.image  # .copy()
        y = 29
        for i in range(6):
            x = i
            x = 23 + (x * 16) + 2 * (x > 2)
            region = (slice(y, y + 10), slice(x - 8, x + 8))
            hsv = image[region]
            sanity_mask = hsv[..., 0] < 5
            hsv[..., 0][sanity_mask] = 179
            wire = -1
            led = image[y - 10, x,2]
            self.leds[i] = WIRE.led.value if  led > 50 else 0
            log.debug(f"LED value-reading: {led} => {self.leds[i]}")
            for color, pairs in wire_ranges.items():
                mask = inRangePairs(hsv, pairs)
                count = np.sum(np.nonzero(mask))
                if count > 150:
                    if wire < color.value:
                        wire = color.value
                    else:
                        wire += color.value
            self.wires[i] = wire
        log.debug(f"leds: {self.leds}")
        log.debug(f"wires: {self.wires}")

class Solver():
    def new(self, robot:RobotArm):
        return ComplexWires(robot)
    def identify(self, robot:RobotArm):
        test = ComplexWires(robot)
        return test.valid * 200, test.debug_image