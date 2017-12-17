from robot_arm import RobotArm
from util import *
import logging
log = logging.getLogger(__name__)
import time
#{'A': '.-', 'B': '-...', 'C': '-.-.',
#'D': '-..', 'E': '.', 'F': '..-.',
#'G': '--.', 'H': '....', 'I': '..',
#'J': '.---', 'K': '-.-', 'L': '.-..',
#'M': '--', 'N': '-.', 'O': '---',
#'P': '.--.', 'Q': '--.-', 'R': '.-.',
#'S': '...', 'T': '-', 'U': '..-',
#'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..'}
# ^^ from the internet
morse_english_dict = {'.-': 'A', '-...': 'B', '-.-.': 'C',
 '-..': 'D', '.': 'E', '..-.': 'F',
 '--.': 'G', '....': 'H', '..': 'I',
 '.---': 'J', '-.-': 'K', '.-..': 'L',
 '--': 'M', '-.': 'N', '---': 'O',
 '.--.': 'P', '--.-': 'Q', '.-.': 'R',
 '...': 'S', '-': 'T', '..-': 'U',
 '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', '--..': 'Z'}
# ^^by the power of regex

#a big block of caps looks ugly in source but we want to match against one ...
words = [str.upper(x) for x in ("shell", "halls", "slick", "trick", "boxes", "leaks", "strobe", "bistro", "flick", "bombs", "break",
         "brick", "steak", "sting", "vector", "beats")]

#letters=set("".join(words))

def from_morse(dots):
    return morse_english_dict.get(dots, f"<{dots}>")

class Morse():
    def __init__(self, robot:RobotArm):
        self.robot = robot
        self.canvas = np.full((150,200,3),120,"uint8")
    #TODO: draw morse

    def draw_letters(self, letters, stage):
        cv2.rectangle(self.canvas, (0,90 + stage * 20), (99,110 + stage * 20),(120,120,120), -1)
        draw_label(self.canvas, (50,100 + stage * 20), ",".join(letters))
        self.robot.draw_module(self.canvas)


    def draw_dots(self,dots):
        cv2.rectangle(self.canvas, (0,60), (99,80),(120,120,120), -1)
        draw_label(self.canvas, (50,70), f"{dots}", font_scale=1)
        self.robot.draw_module(self.canvas)

    def draw_state(self, state):
        color = (20,230,230) if state else (120,120,120)
        c2 = (20,180,180) if state else (120,120,120)
        cv2.circle(self.canvas, (50,20), 15, c2, -1)
        cv2.rectangle(self.canvas, (20,10), (80,30), color)
        self.robot.draw_module(self.canvas)
    def solve(self):
        solved, solution = self.read_word()
        if not solved:
            solved, solution = self.read_word(partial=solution)
        if not solved:
            return False
        log.info(f"Word is: {solution}, changing frequency")
    #word list is in order, so we just need to click on the > button a number of times equal to its index in the array
        self.robot.moduleto(128,90)
        for i in range(words.index(solution)):
            self.robot.click(after=0.1)
        log.info("responding")
        self.robot.moduleto(80,126)
        self.robot.click(after=0.1)
        return True

    def get_state(self):
        im = self.robot.grab_selected()
        led = im[20:30,42:70]
        return np.sum(to_hsv(led)[...,2] > 150) > 200

    def read_word(self, partial = None):
        letters = ""
        possibles = [p for p in words if p.endswith(partial)] if partial else words.copy()
        log.debug(f"retrying with letters {partial} | {possibles}")
        if partial is None:
            #trash the first letter - we might have started listening half way through it
            trash = self.read_dots()
        else:
            if len(possibles) == 1:
                log.debug(f"only one possible match for ...{partial}: {possibles[0]}")
                return True, possibles[0]

            #drop the whole end-of-word pause
            trash = self.read_statechange()
        while "reading letters":
            dots = self.read_dots()
            if not len(dots):
                log.debug("end-of-word read with no word decided")
                return False, letters
            else:
                letter = from_morse(dots)
                log.debug(f"{dots} - > {letter}")
                letters += letter
                self.draw_letters(letters, partial is None)
                if partial is None:
                    possibles = [p for p in possibles if letters in p]
                else:
                    possibles = [p for p in possibles if p.startswith(letters)]
                log.debug(f"{len(possibles)} matches for {letters}")
                if len(possibles) == 0:
                    #this shouldn't happen unless we've misread something
                    log.debug(f"ran out of possible matches for {letters}, aborting")
                    return False, letters
                if len(possibles) == 1:
                    mask = letters
                    if partial is not None:
                        mask = f"{letters} ... {partial}"
                    log.debug(f"only one possible match for {mask}: {possibles[0]}")
                    return True, possibles[0]

    
    def read_dots(self):
        dots = ""
        while "Reading Letter":
            state, t_diff = self.read_statechange(timeout=1)
            if state:
                dots += "." if t_diff < 0.5 else "-"
                self.draw_dots(dots)
            else:
                if t_diff > 0.45:
                    return dots

    def read_statechange(self, previous_state = None, timeout=4):
        """Watches the light and returns when it changes state or hits the timeout
        returns previous state, delta-time"""
        if previous_state is None: previous_state = self.get_state()
        self.draw_state(previous_state)
        starttime = time.time()
        while "Reading state":
            state  = self.get_state()
            t_diff = time.time()-starttime
            if not (state == previous_state):
                self.draw_state(state)
                return previous_state, t_diff
            if 0 < timeout <= t_diff :
                    return 0, t_diff



                #if not keep_showing(im):
                #   break

class Solver():
    def __init__(self):
        pass


    def new(self, robot:RobotArm):
        return Morse(robot)

    def identify(self, robot:RobotArm):
        im = robot.grab_selected()
        hsv = to_hsv(im)
        mask = inRangePairs(hsv, [(5, 26), (149, 215), (0, 255)])
        canvas = to_bgr(gray=mask)
        cnts = contour(mask, mode=cv2.RETR_EXTERNAL)
        for cnt in cnts:
            x,y, w,h = cv2.boundingRect(cnt)
            #if it's in the right area and the right size
            if 38 < x < 48 and 8 < y < 16 and 20 < w < 30 and 14 < h < 22:
                cv2.drawContours(canvas, [cnt],-1, (0,255,0),2)
                return 100, canvas
            else:
                cv2.drawContours(canvas, [cnt], -1, (0, 0, 255), 2)
        return False, canvas
