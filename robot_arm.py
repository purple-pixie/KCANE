from pynput.mouse import Button, Controller
from pynput.keyboard import Key, Listener
import time
from enum import Enum
import cv2
import threading

mouse = Controller()

def goto():
    pass

def on_press(key):
    if key == Key.space:
        return False
    return True

def on_release(key):
    return True

class state(Enum):
    INACTIVE = -1
    NEUTRAL = 0
    ACTIVE = 1

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

def sleep(dur: float, abort=True):
    #seconds to milis
    if abort:
        if threading.active_count() < 2:
            print(f"key: {threading.active_count()} aborting")
            raise KeyboardInterrupt("User aborted")
    time.sleep(dur)


class RobotArm:
    def __init__(self, screen, wake_up = False):
        self.screen = screen
        self.state = state.NEUTRAL
        listen = Listener(on_press=on_press, on_release=on_release)
        listen.start()
        if wake_up:
            self.mouseto(100, 100)
            print("Unselecting module / dropping bomb")
            self.rclick(after = 0.5)
            print("Dropping bomb")
            self.rclick(after = 0.5)
            self.state = state.INACTIVE
            self.pick_up()
            self.state = state.ACTIVE
    def examine_gubbins(self):
        self.mouse_to_centre()
        for i in range(4):
            self.rotate(i)
            #self.screen.save_screen()
            yield self.screen.grab()
            self.unrotate()


    def rotate(self, dir = 0):
        mouse.press(Button.right)
        #allow game to realise we've clicked
        sleep(0.05)
        x, y = self.mouse_position()
        #self.mouseto(x + (left and -100 or 100), y)
        dir = dir % 4
        if dir == 0: x = x - 150
        elif dir == 1: y = y - 150
        elif dir == 2: x = x + 150
        elif dir == 3: y = y + 150
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

    def unrotate(self, dir = 0):
        self.mouse_to_centre()
        sleep(0.5)
        mouse.release(Button.right)
        sleep(0.05)

    def pick_up(self):
        self.mouse_to_centre()
        self.click(0, 0.5)

    def mouse_position(self):
        return self.screen.scaled_coords(*mouse.position)

    def mouse_to_centre(self):
        self.mouseto(410, 295)
    def mouseto(self, x, y):
        mouse.position = self.screen.real_coords(x, y)
        print(f"mouse now at {mouse.position}")
    def goto(self, mod):
        self.mouseto(*modules[mod])
        self.click(0, 1)
        #move to the middle of the focus tile
        #self.mouse_to_centre()
    def moduleto(self, x, y):
        self.mouseto(x + selected_left, y + selected_top)
    def grab_selected(self, colour = True):
        """grab a screenshot of the active module"""
        im = self.screen.grab()
        im = im[selected_top:selected_bot,
               selected_left:selected_right]
        if not colour:
            return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return im

    def click(self, before=0, after=0, button=Button.left):
        mouse.press(button)
        if before:
            sleep(before)
        mouse.release(button)
        if after:
            sleep(after)

    def rclick(self, **kwargs):
        self.click(button=Button.right, **kwargs)