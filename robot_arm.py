from pynput.mouse import Button, Controller
from pynput.keyboard import Key, Listener
import time
from enum import Enum
import cv2
import threading
import bomb_examiner
from util import *
mouse = Controller()


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
    def __init__(self, screen, wake_up = False, window_name = None):
        self.screen = screen
        self.window_name = window_name
        #listen for the user pressing space
        #if they ever do, quit out next time we're asked to sleep
        listen = Listener(on_press=on_press, on_release=on_release)
        listen.start()
    def wake_up(self):
        self.mouseto(100, 100)
        #print("Unselecting module / dropping bomb")
        self.rclick(after = 0.25)
        #print("Dropping bomb")
        self.rclick(after = 0.25)
        self.pick_up()

    def panic(self, dir = "screens/"):
        self.screen.save_screen(dir)

    def scan_modules(self, dump = False):
        for mod in range(6):
            self.mouse_to_module(mod)
            sleep(0.1, True)
            s = self.grab()
            if dump: dump_image(s, starts=f"mod{mod}_")
            yield bomb_examiner.find_highlight(s)

    def get_edges(self):
        for i in range(4):
            self.rotate(i)
            #self.screen.save_screen()
            yield self.grab()
            self.unrotate()


    def rotate(self, dir = 0):
        """rotate the bomb - does not let go of right click. Do that yourself
        dir 0-3 rotate 90 degrees left, up, right, down
        dir 4 and 5 rotate 180 degrees left and right"""
        self.mouse_to_centre()
        mouse.press(Button.right)
        #allow game to realise we've clicked
        sleep(0.05)
        x, y = self.mouse_position()
        #self.mouseto(x + (left and -100 or 100), y)
        #dir = dir % 4
        if dir == 0: x = x - 150
        elif dir == 1: y = y - 150
        elif dir == 2: x = x + 150
        elif dir == 3: y = y + 150
        elif dir == 4: x = x - 300
        elif dir == 5: x = x + 300
        else:
            print(f"dir {dir} not sane (RobotArm.rotate)")
            return
            #mouse.move(int(150 / self.screen.xscale), 0)

            #rotate left:
            #self.mouseto(x-150, y)

            #self.mouseto(x, y-132)
        self.mouseto(x, y)
        #allow bomb to actually rotate on screen
        sleep(0.5)
        if dir > 3:
            mouse.release(Button.right)
            sleep(0.1)

    def unrotate(self, dir = 0):
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
        mouse.position = self.screen.real_coords(x, y)
        #log#print(f"mouse now at {mouse.position}")
    def mouse_to_module(self, mod):
        self.mouseto(*modules[mod])
    def goto(self, mod, after = 1):
        self.mouse_to_module(mod)
        self.click(0, after)
        #move to the middle of the focus tile
        #self.mouse_to_centre()
    def moduleto(self, x, y):
        self.mouseto(x + selected_left, y + selected_top)

    def grab(self, colour = True):
        #todo: lights out
        im = self.screen.grab()
        if not colour:
            if len(im.shape) > 2:
                return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return im
    def grab_selected(self, colour = True):
        """grab a screenshot of the active module"""
        im = self.grab(colour)
        if im.shape[0] >= selected_bot:
            im = im[selected_top:selected_bot,
                   selected_left:selected_right]
        return im

    def click(self, before=0., after=0., button=Button.left):
        mouse.press(button)
        if before:
            sleep(before)
        mouse.release(button)
        if after:
            sleep(after)
    def rclick(self, **kwargs):
        self.click(button=Button.right, **kwargs)

    def draw_module(self, image):
        im=self.grab()
        canvas = im[selected_top:selected_bot,
                   selected_left:selected_right]
