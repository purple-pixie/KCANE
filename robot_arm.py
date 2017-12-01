from pynput.mouse import Button, Controller
from pynput.keyboard import Key, Listener
import time
from enum import Enum
import cv2
import threading
import bomb_examiner
import clock
from util import *
mouse = Controller()

import logging
log = logging.getLogger(__name__)

def sleep(dur: float, abort=True):
    # seconds to milis
    if abort:
        # a pretty hacky solution but it detects keypresses
        # the keyboard listener will quit once it sees a letter, so <2 threads means it quit
        # running the debugger makes it never abort, but you're debugging so that's probably good
        if threading.active_count() < 2:
            print(f"key: {threading.active_count()} aborting")
            mouse.release(button=Button.right)
            raise KeyboardInterrupt("User aborted")
    time.sleep(dur)

def on_press(key):
    if key == Key.space:
        return False

    return True

def on_release(key):
    return True

top = 230
bot = 380
left = 250
width = 150

modules = ([[left, top], [left + width, top], [left + width + width, top],
            [left, bot], [left + width, bot], [left + width + width, bot]])

#[220:378,330:496]
selected_top = 220
selected_bot = 378
selected_left = 330
selected_right = 496


class RobotArm:
    def __init__(self, screen, wake_up = False, robot = None):
        self.screen = screen
        self.robot = robot
        self.selected = -1
        #listen for the user pressing space
        #if they ever do, quit out next time we're asked to sleep
        listen = Listener(on_press=on_press, on_release=on_release)
        listen.start()
    def wake_up(self):
        self.mouseto(100, 100)
        log.debug("Unselecting module / dropping bomb")
        self.rclick(after = 0.25)
        log.debug("Ensuring bomb is dropped")
        self.rclick(after = 0.25)
        self.pick_up()


    def indicator_state(self, image = None):
        """the solved state of the indicator on the currently selected bomb
        (or from image)
        assumes a currently selected module
        1 for solved, -1 for red (warning light) or 0 for unlit"""
        if image is None:
            image = self.grab_selected(allow_dark=True)
        region = image[:23 , 130:]
        indicator = to_hsv(region)
        green = inRangePairs(indicator, [(58, 74), (222, 255), (204, 251)])
        if np.sum(green) > 2550:
            #log.debug("it is solved")
           # dump_image(region, starts="green", dir="indicator")
            return 1
        red = inRangePairs(indicator, [(167, 179), (175, 255), (208, 255)]) # red indicator#
        if np.sum(red) > 2550:
            #dump_image(region, starts="red", dir="indicator")
            return -1
       # dump_image(region, starts="none", dir="indicator")
        return 0

    def draw(self):
        if not self.robot is None:
            self.robot.draw()
    def panic(self, dir = "screens/"):
        self.screen.save_screen(dir)

    def scan_modules(self, dump = False):
        for mod in range(6):
            image = self.grab_module_unfocused(mod)
            #self.mouse_to_module(mod)
            #sleep(0.1, True)
            #s = self.grab()
            if dump: dump_image(image, starts=f"mod{mod}_")
            #find the clock?
            #can probably just let simple buttons deal with it themselves
            if bomb_examiner.is_empty(image):
                yield "empty"
            else:
                yield "clock" if clock.isClock(image) else "unidentified"

    def get_edges(self):
        for i in range(4):
            self.rotate(i)
            #self.screen.save_screen()
            yield i, self.grab()
            self.unrotate()


    def rotate(self, dir = -1, x=0, y=0):
        """rotate the bomb - does not let go of right click. Do that yourself
        dir 0-3 rotate 90 degrees left, up, right, down
        dir 4 and 5 rotate 180 degrees left and right"""
        if self.robot.safe: return
        self.mouse_to_centre()
        mouse.press(Button.right)
        #allow game to realise we've clicked
        sleep(0.05)
        if not dir == -1:
            x, y = self.mouse_position()
            #self.mouseto(x + (left and -100 or 100), y)
            #dir = dir % 4
            if dir == 0: x = x - 146
            elif dir == 1: y = y - 146
            elif dir == 2: x = x + 146
            elif dir == 3: y = y + 148
            elif dir == 4: x = x - 292
            elif dir == 5: x = x + 292
            else:
                print(f"dir {dir} not sane (RobotArm.rotate)")
                return
            #mouse.move(int(150 / self.screen.xscale), 0)

            #rotate left:
            #self.mouseto(x-150, y)

            #self.mouseto(x, y-132)
        self.mouseto(x, y)
        #allow bomb to actually rotate on screen
        sleep(0.2)
        if dir > 3:
            mouse.release(Button.right)
            sleep(0.2)

    def unrotate(self):
        if self.robot.safe:return
        self.mouse_to_centre()
        sleep(0.5)
        mouse.release(Button.right)
        sleep(0.05)

    def pick_up(self):
        self.mouse_to_centre()
        self.click(0, 0.25)

    def mouse_position(self):
        return self.screen.scaled_coords(*mouse.position)

    def mouse_to_centre(self):
        self.mouseto(410, 295)
    def mouseto(self, x, y):
        if self.robot.safe:return
        mouse.position = self.screen.real_coords(x, y)
        #log#print(f"mouse now at {mouse.position}")
    def mouse_to_module(self, mod):
        self.mouseto(*modules[mod])


    def goto_from(self, desired, after=0.3):
        """goto module desired, starting from self.selected"""
        current = self.selected
        c_x, c_y = current%3,current//3
        d_x, d_y = desired%3, desired//3
        x = 400 + (d_x-c_x) * 200
        y = 300 + (d_y-c_y) * 200
        x = min(max(x, 75), 725)
        self.mouseto(x, y)
        self.click(0, after)
        self.selected = desired

    def goto(self, mod, after = 1):
        self.mouse_to_module(mod)
        self.click(0, after)
        self.selected = mod
        #move to the middle of the focus tile
        #self.mouse_to_centre()
    def moduleto(self, x, y):
        self.mouseto(x + selected_left, y + selected_top)

    def grab(self, colour = True, allow_dark = False, allow_red = False):
        #todo: lights out / red light check
        im = self.screen.grab()
        if not colour:
            if len(im.shape) > 2:
                return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return im



    def grab_other_module(self, desired):
        """grab module at pos, assuming we are looking at current and want desired"""
        current = self.selected
        c_x, c_y = current%3,current//3
        d_x, d_y = desired%3, desired//3

        pad_x = selected_right - selected_left + 25
        pad_y = selected_bot - selected_top + 20

        x1 = selected_left + pad_x * (d_x-c_x)
        x2 = selected_right+ pad_x * (d_x-c_x)
        y1 = selected_top + pad_y * (d_y - c_y)
        y2 = selected_bot + pad_y * (d_y - c_y)

        return self.grab()[y1:y2, x1:x2]

    def grab_module_unfocused(self, pos):
        """grab module at position pos (0-5) assuming no module currently selected"""
        x,y = pos%3, pos//3
        y1 = 160
        y2 = 298
        x1 = 180
        x2 = 316
        h = y2-y1
        w = x2-x1
        return self.grab()[y1 + y * h + y * 12:y2 + y * h + y * 12, x1 + x*w + x*25 : x2+x*w+x*25]

    def grab_selected(self, colour = True, allow_dark = False, allow_red = False):
        """grab a screenshot of the active module"""
        im = self.grab(colour, allow_dark, allow_red)
        #standard case - grab the region
        if im.shape[0] >= selected_bot:
            im = im[selected_top:selected_bot,
                   selected_left:selected_right]
        #special case for loading screens that are already cropped to selected region
        #(or at least to the right height, assume there might be more junk hstack()ed on afterwards
        if im.shape[0] == selected_bot-selected_top:
            im = im[:,:selected_right-selected_left]
        return im

    def click(self, before=0., after=0., button=Button.left, between = 0., dir = None):
        if self.robot.safe: return sleep(after)
        if not dir is None:
            dump_image(self.grab_selected(),dir=dir,starts="pre_sleep")
        if before:
            sleep(before)
        if not dir is None:
            dump_image(self.grab_selected(),dir=dir,starts="pre_click")
        mouse.press(button)
        if between:
            sleep(between)
        mouse.release(button)
        if after:
            sleep(after)
        if not dir is None:
            dump_image(self.grab_selected(),dir=dir,starts="post_click")
    def rclick(self, **kwargs):
        self.selected = -1
        self.click(button=Button.right, **kwargs)

    def draw_module(self, image):
        self.robot.draw_module(image)
