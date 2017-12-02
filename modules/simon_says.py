from util import *
import logging
import robot_arm
log = logging.getLogger(__name__)
import time
from enum import Enum
#v-min 220 catches lit square

# sat 150+ - only the 4 squares
# (175+ is cleaner but puts a hole in the yellow)


#ys - 35, 62
class COLOR(Enum):
    red=0
    blue=1
    green=2
    yellow=3

b = COLOR.blue
y = COLOR.yellow
g = COLOR.green
r = COLOR.red
moves = [
            ([b,y,g,r], [r,b,y,g], [y,g,r,b]),
            ([b,r,y,g], [y,g,b,r], [g,r,y,b])
         ]
    
button_regions = [(35,60), (70,30), (70, 90), (105,60) ]
button_hues = [179, 118,  54, 25]

class SimonSays():
    def __init__(self, robot:robot_arm.RobotArm):
        self.robot:RobotArm = robot
        return

    def get_lit_button(self, timeout=5.):
        #reading a lit button works in the dark and in redlight

        #give up after 5 seconds, it shouldn't take that long
        starttime = time.time()
        while time.time() - starttime < timeout:
            hsv = to_hsv(self.robot.grab_selected(allow_dark=True,allow_red=True))
            sat_mask = hsv[...,1] < 150
            hsv[sat_mask] = (0,0,0)
            for c, (x,y) in enumerate(button_regions):
                vmean = np.mean(hsv[y:y+20,x:x+20,2])
                if vmean > 150:
                    #log.debug(f"{COLOR(c)} highlit (vmean: {vmean})")
                    return COLOR(c)
                #log.debug(f"{COLOR(c)} | {vmean}")
                # cv2.imwrite("s.bmp",s)
                #display(to_bgr(hsv), wait_forever=False)
        if timeout > 1:
            log.debug(f"timed out waiting for a lit button")
        return None

    def watch(self):
        """stare at the bomb and dump useful looking stuff to the console
        for timing research / debugging"""
        t_start = time.time()
        while 'main loop':
            btn = self.get_lit_button(0.5)
            while btn is not None:
                #if something is already lit, wait for it to go out
                print(f"{btn} is still lit")
                btn = self.get_lit_button(0.5)
            btn = self.get_lit_button(5)
            if btn is None:
                print("No lit found, abort")
                continue
            first = btn
            print(f"Cycle begins: {first} | {time.time() - t_start}")
            while "cycle loop":
                #cycle logic
                button = self.get_lit_button(0.1)
                print(f"{button}->", end="")
                while button is not None:
                    print(f"+",end="")
                    button = self.get_lit_button(0.1)
                button = self.get_lit_button(1)
                if button is None:
                    break
                print()



            print(f"Cycle ends | {time.time() - t_start}")

            cv2.waitKey(1)
        pass

    def translate(self, color:COLOR):
        """get the corresponding color for the given color"""
        vowel = self.robot.robot.serial_has_vowel()
        strikes = self.robot.robot.get_strikes()
        #vowel, strikes = 1, 1
        log.debug(f"vowel: {vowel}, {strikes} strikes | {color.name}->")
        target = moves[vowel][strikes > 0][color.value]
        log.debug(f"{target.name}")
        return target

    def draw(self, stage, seq):
        y = stage * 30 + 10
        for i, color in enumerate(seq):
            x = i * 30 + 10
            col = (0,0,255)
            if color == COLOR.blue:
                col = (255,0,0)
            elif color == COLOR.yellow:
                col = (0,255,255)
            elif color == COLOR.green:
                col = (0,255,0)
            cv2.rectangle(self.canvas, (x,y), (x+20,y+20), col, -1)
        self.robot.draw_module(self.canvas)

    def solve_stage(self, steps):
        sequence = []
        for i in range(steps+1):
            #after each stage except the last (i.e. before every stage except the first_
            #wait for the last light to go out
            if i > 0:
                while self.get_lit_button(0.1) is not None:
                    pass
                button = self.get_lit_button(1)
            else:
                button = self.get_lit_button()
            log.info(f"solving stage {steps}-{i}: {button}")
            if button is None:
                #will get here if we lose track in the middle of a loop too.
                #solver should try once more on the same stage
                return None
            sequence.append(self.translate(button))

            #give it long enough to go out again
        self.draw(steps, sequence)
        log.debug(f"Sequence: {sequence}")
        for i, color in enumerate(sequence):
            self.robot.moduleto(*button_regions[color.value])
            self.robot.click(before=0.1 if i else 0.5) #, after=0.1)
        return True
    def solve(self, retry = True):
        self.canvas = np.full((170, 170, 3), 120, dtype="uint8")
        #TODO: remember that the solution should match the last stage plus a move
        #or ditch the whole stage thing? could pick up half solved runs
        for stage in range(6):
            if self.solve_stage(stage) is None:
                if not self.solve_stage(stage):
                    break
            log.debug(f"solve pre-stage nap")
            robot_arm.sleep(0.2)
            indicator = to_hsv(self.robot.grab_selected()[:23 , 130:])
            green = inRangePairs(indicator, [(58, 74), (222, 255), (204, 251)])
            red = inRangePairs(indicator, [(167, 179), (175, 255), (208, 255)]) # red indicator#
            if np.sum(green) > 2550:
                log.debug("woo, it's solved")
                return True
            if np.sum(red) > 2550:
                if retry:
                    log.info("I screwed up. Assuming the serial was wrong and restarting.")
                    self.robot.robot.serial_vowel_error()
                    robot_arm.sleep(2)
                    return self.solve(retry=False)
                else:
                    log.info("I really screwed up, aborting")
                    return False
            log.debug(f"solve post-stage sleep")
            robot_arm.sleep(.5)
        else:
            return True
        return False

class Solver():
    def __init__(self):
        pass
    def new(self, robot:robot_arm.RobotArm):
        x = SimonSays(robot)
        return x
    def identify(self, robot:robot_arm.RobotArm):
        #TODO - Simon seems to be missing idents when a button is lit.
        #check for that explicitly or use a heuristic that allows for it
        image = robot.grab_selected()
        #dump_image(image)
        hsv  = to_hsv(image)
        h, s, v = (hsv[...,i] for i in range(3))
        #sanity adjust hue so that reds are all together:
        sanity_mask = h < 5
        h[sanity_mask] = 178
        for c, ((x, y), hue) in enumerate(zip(button_regions, button_hues)):
            #x, y = button_regions[color]
            #hue = button_hues[c]
            cv2.rectangle(image, (x, y), (x+20, y+20), (0,0,0),3)
            diff = abs(hue-np.mean(h[y:y+20,x:x+20]))
            if diff > 15:
                color = COLOR(c).name
                log.debug(f"{color} button doesn't look very {color} (diff: {diff})")
                return False, image
        return 200, image


