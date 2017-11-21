import numpy as np
import cv2
import mss
import glob
from util import *

screen_width = 800
screen_height = 600
class Screen:
    def __init__(self, monitor = 0, image = None):
        if not image is None:
            self.sct = None

            ###hacky hacky type testing. We don't need no stupid alpha channel
            if len(image.shape) > 2 and image.shape[2] == 4:
                    self.image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                self.image = image.copy()
            #/hack

            self.xscale = 1
            self.yscale = 1
            with mss.mss() as a:
                self.monitor = dict(a.monitors[2])
            return
        self.image = None
        if monitor != 2:
            print("Not currently sure how to deal with monitors other than 2")
            raise IOError("Not currently sure how to deal with monitors other than 2")
        self.sct = mss.mss()
        monitor = dict(self.sct.monitors[monitor])
        print(self.sct.monitors[2])
        image = np.array(self.sct.grab(monitor))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cols = np.sum(image, axis=0)
        xbuffer = np.count_nonzero(cols == 0)
        self.x = 0
        if xbuffer == image.shape[1]:
            print("Screen is all black")
            raise IOError("Black Screen")
            #image is all black, don't divide by 0
        else:
            monitor["left"] += int(xbuffer / 2)
            monitor["width"] -= xbuffer
            self.xscale = screen_width / (image.shape[1] - xbuffer)
        self.yscale = screen_height / image.shape[0]
        self.monitor = monitor
    def scaled_coords(self, x, y):
        return int((x - self.monitor["left"]) * self.xscale), int((y - self.monitor["top"]) * self.yscale)
    def real_coords(self, x, y):
        return int(x / self.xscale) + self.monitor["left"], int(y / self.yscale) + self.monitor["top"]
    def grab(self):
        if not self.image is None:
            return self.image.copy()
        image = self.convert(np.array(self.sct.grab(self.monitor)))
        if len(image.shape) > 2 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
    def convert(self, image):
        if self.xscale != 1 or self.yscale != 1:
            image = cv2.resize(image, (screen_width, screen_height), interpolation=cv2.INTER_AREA)
        return image

    def save_screen(self, dir = "screen/"):
        im = self.grab()
        dump_image(im, dir)

    def __repr__(self):
        return f"Screen object, coords {self.monitor}"