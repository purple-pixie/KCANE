#mask = inRangePairs(im, ((0, 0), (0, 0), (255, 255)))
from util import *
import robot_arm
import logging
log = logging.getLogger(__name__)

ranges = [
    ("yellow", [(19, 33), (192, 255), (149, 255)]),
    ("white", [(0, 33), (0, 255), (137, 255)] ),
    ("blue", [(110, 120), (160, 255), (97, 255)]),
    ("red", [(174, 179), (73, 255), (163, 255)])]

def get_letters(mask):
    mask = cv2.dilate(mask, (5, 5), iterations=1)
    #display(mask)
    cnts = zip(*contour(mask, mode=cv2.RETR_TREE, return_hierarchy=True))
    cnt, _ = next(cnts)
    if not 300 < cv2.arcLength(cnt, True) < 340:
        return ()
    cnts = list(cnts)
    for cnt, data in cnts:
        if data[3] == 0:
            yield Contour(cnt)


def get_color_letters(image):
    im = to_hsv(image)
    for color, pairs in ranges:
        mask = inRangePairs(im, pairs)
        # if there aren't a lot of pixels matched it's not a button, don't waste time on contours
        if np.sum(mask) < 1000000:
            continue
        letters = list(get_letters(mask))
        return color, letters
    return None, None

class Button():
    def __init__(self, robot:robot_arm.RobotArm):
        self.robot = robot
        self.text = "NONE"
        im = robot.grab_selected()
        self.color, letters = get_color_letters(im)
        if letters is None:
            return
        if len(letters) == 4:
            self.text = "HOLD"
        if len(letters) == 8:
            self.text = "DETONATE"
        if len(letters) == 5:
            self.text = "PRESS" if len(letters[0]) > 25 else "ABORT"

    def solve(self):
        #DEBUG this is why buttons don't solve atm
        log.debug(f"defusing | {self}")
        return False
        release = self.is_press_release()
        if release:
            print("Press and Release")
            self.robot.moduleto(60,60)
            self.robot.click(before=0.1, between=0.1)
        else:
            print("Press and hold")
            clock = self.robot.robot.get_clock_reading()
            print(f"#### clock; {clock} ####")
            self.robot.mouse_to_centre()
            x, y = self.robot.mouse_position()
            while clock == "":
                self.robot.rotate(x=x+random.randrange(-30,30,2), y=y)
                clock = self.robot.robot.get_clock_reading()
                print(".",end="",sep="")
            #TODO: actually solve this



    def is_press_release(self):
        if self.color == "blue" and self.text == "ABORT":
            return False
        if self.robot.robot.battery_count > 1 and self.text == "DETONATE":
            return True
        if self.color == "white" and self.robot.robot.has_indicator("CAR"):
            return False
        if self.robot.robot.battery_count > 2 and self.robot.robot.has_indicator("FRK"):
            return True
        if self.color == "yellow":
            return False
        if self.color == "red" and self.text == "HOLD":
            return True
        return False



    def __repr__(self):
        return f"Button | {self.color} {self.text}"

class Solver():
    def identify(self, robot:robot_arm.RobotArm):
        canvas = robot.grab_selected()
        color, letters = get_color_letters(canvas)
        if color is None:
            return False, None
        for cnt in letters:
            cnt.draw(canvas, rect_color=(0, 255, 20), detail_color=(0, 0, 255))
        if len(letters) in (4,5,8):
            return 200, canvas
        return 60 if 2 < len(letters) < 10 else False, canvas


    def new(self, robot:robot_arm.RobotArm):
        return Button(robot)

