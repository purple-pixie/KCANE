#utility functions for KCANE
#mostly just shorthand for very regularly called functions

import cv2
from PIL import Image
import pytesseract
import os
import random
import glob
from pathlib import Path
import threading
import time
import sys
import inspect

class PrintSnooper:
    def __init__(self, stdout):
        self.stdout = stdout
    def caller(self):
        return inspect.stack()[2][3]
    def write(self, s):
        self.stdout.write("printed by %s: " % self.caller())
        self.stdout.write(s[:100])
        self.stdout.write("\n")

def images_in(dir = "", flags=1, ext = ".bmp"):
    for im in glob.glob(f"{dir}*{ext}"):
        yield cv2.imread(im, flags)

def read_key(directions = True):
    key = 256
    while key > 127:
        key = cv2.waitKeyEx(0)
        if directions:
            print(key)
            if key in ( 2490368,
                        2621440,
                        2424832,
                        2555904,
                        ):  # up, down, left, right
                return key
    return key&0xFF

def display(c, text="image", do_wait = True, scale = None):
    if not scale is None:
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(text, c)
    if do_wait:
        cv2.waitKey(0)
        #cv2.destroyAllWindows()

def read_and_display(*args, **kwargs):
    kwargs["do_wait"] = False
    display(*args, **kwargs)
    return read_key()


def keep_showing(x, save_func = None):
    """show an image and return whether user pressed Q or not
    if they did, return False (i.e. caller should not keep showing), else True (keep showing)
    if save_func is provided and user presses S, call save_func on the image"""
    cv2.imshow("screen",x)
    key = cv2.waitKey(25) & 0xFF
    if  key == ord('q'):
        cv2.destroyAllWindows()
        return False
    if save_func and key == ord("s"):
            save_func(x)
    return True


def naked_filename(path):
    return (Path(path).resolve().stem)

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def otsu(image):
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image

def tess(image, config="--psm 10 digits", remove = True):
    filename = "tess_temp.bmp"
    cv2.imwrite(filename , image)
    out = pytesseract.image_to_string(Image.open(filename ), config=config)
    if remove:
        os.remove(filename)
    return out

def contour(image):
    (im, contours, hierarchy) = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_dump_name(dir="", ext="bmp", starts=""):
    if "." in ext:
        ext = ext.lstrip(".")
    if starts == "":
        starts = "dump"
    starts = os.path.join("dump",dir, starts)
    hits = 0
    path = f"{starts}{hits}.{ext}"
    while os.path.isfile(path):
        hits += 1
        path = f"{starts}{hits}.{ext}"
    return path


def dump_image(img, dir="test/", announce = True, starts = "img"):
        filename = get_dump_name(dir, "bmp", starts)
        if announce:
                print(f"dumping to {filename}")
        cv2.imwrite(filename, img)

def sleep(dur: float, abort=True):
    # seconds to milis
    if abort:
        if threading.active_count() < 2:
            print(f"key: {threading.active_count()} aborting")
            raise KeyboardInterrupt("User aborted")
    time.sleep(dur)
