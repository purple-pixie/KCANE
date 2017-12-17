#utility functions for KCANE
#mostly just shorthand for very regularly called functions
import collections
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
import numpy as np

class StrikeException(Exception):
    """Raised when a solver causes a strike"""
    pass

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

class PrintSnooper:
    def __init__(self, stdout):
        self.stdout = stdout
    def caller(self):
        return inspect.stack()[2][3]
    def write(self, s):
        self.stdout.write("printed by %s: " % self.caller())
        self.stdout.write(s[:100])
        self.stdout.write("\n")
#sys.stdout = PrintSnooper(sys.stdout)
def images_in(dir = "", flags=1, ext = ".bmp", return_names = False, return_full_names = False,
              dont_open = False, starts = ""):
    starts = os.path.join(dir, starts)
    for name in glob.glob(f"{starts}*{ext}"):
        im = None if dont_open else cv2.imread(name, flags)
        if return_full_names: yield name, im
        elif return_names: yield naked_filename(name), im
        else: yield im

def read_key(directions = True):
    key = 256
    while key > 127:
        key = cv2.waitKeyEx(0)
        if directions:
            if key in ( 2490368,
                        2621440,
                        2424832,
                        2555904,
                        ):  # up, down, left, right
                return key
    return key&0xFF
def hstack_pad(a, b=None):
    if b is None:
        return a
    if a is None or a.shape[0] == 0:
        return b.copy()
    if a.shape[0] == b.shape[0]:
        return np.hstack((a,b))
    if a.shape[0] <= b.shape[0]:
        return hstack_pad(b, a)
    pad = a.shape[0] - b.shape[0]
    padded = np.pad(b,((pad,0),(0,0), (0,0)),mode="constant")
    return np.hstack((a,padded))


def display(c, text="image", wait_forever = True, scale = None):
    if not scale is None:
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(text, c)
    cv2.waitKey(0 if wait_forever else 1)

def read_and_display(*args, **kwargs):
    kwargs["wait_forever"] = False
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

def consume(iterable):
    collections.deque(iterable, maxlen=0)

def circles(cnts, key = lambda c: True):
    """return all circles in """
    for cnt in cnts:
        c = cv2.minEnclosingCircle(cnt)
        (a, b), r = c
        if key(c):
            yield (int(x) for x in (a,b,r))

def rotated(image, angle):
    center = tuple(np.array(image.shape[:2])/2)
    M = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(image, M, image.shape[:2],flags=cv2.INTER_LINEAR)

def rectangles(cnts, key = lambda x,y,w,h: True):
    """get all the bounding rectangles of the contours in cnts that satisfy filter key(*rect)"""
    for cnt in cnts:
        rect = cv2.boundingRect(cnt)
        if key(*rect):
            yield rect
def blur(image, kernel=(3,3)):
    return cv2.GaussianBlur(image, kernel, 0)
def contour(image, is_color=False, blur_kernel=None, draw = False,
            return_hierarchy = False, mode=cv2.RETR_LIST, **kwargs):
    if is_color:
        if draw:
            can = image.copy()
        image = to_gray(image)
    else:
        if draw:
            can = to_bgr(gray=image)
    if blur_kernel is not None:
        image = cv2.GaussianBlur(image, blur_kernel, 0)
    (im, contours, hierarchy) = cv2.findContours(image,mode,cv2.CHAIN_APPROX_SIMPLE, **kwargs)
    if draw:
        cv2.drawContours(can, contours, -1, (0,255,0),1)
        display(image)
        display(can)
        #for c in contours:
        #cv2.im
    if return_hierarchy:
        if hierarchy is not None:
            return contours, hierarchy[0]
        else:
            return contours, []
    return contours

def draw_rect(image, x, y, w, h, color=(0,0,0), thickness=1, draw_centred = False):
    if draw_centred:
        x, y = x-w//2, y-h//2
    cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness)

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

def draw_label(image, centre=None, label="<label>", color = (0,0,0), font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=.5, thickness=1):
    """Draws label for point to a given image.
    Returns copy of image, original is not modified.
    """
    # http://docs.opencv.org/modules/core/doc/drawing_functions.html#gettextsize
    # Returns bounding box and baseline -> ((width, height), baseline)
    size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    if centre is None:
        centre = image.shape[0] // 2, image.shape[1] // 2
    x, y = centre
    label_top_left = (x - size[0] // 2, y + size[1] //2 )
    cv2.putText(image, label, label_top_left, font, font_scale, color, thickness)

def inRangePairs(im, pairs):
    ranges = np.array(pairs, dtype="uint8")
    ranges = ranges.transpose()
    return inRange(im, *ranges)
def inRange(im, lower, upper):
    l=np.array(lower, dtype="uint8")
    u = np.array(upper, dtype="uint8")
    return cv2.inRange(im, l, u)

def to_hsv(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
def to_bgr(hsv=None, gray = None):
    if not gray is None:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

class Contour():
    def __init__(self, cnt):
        self.points = cnt
        self.box = None
        self.length = None
    def draw(self, image, detail_color = (0,0,255), rect_color = None, detail_width = 1):
        if rect_color is not None:
            cv2.rectangle(image, (self.x-1, self.y-1), (self.x+self.w, self.y+self.h), rect_color, 1)
        if detail_color is not None:
            cv2.drawContours(image, [self.points], -1, detail_color, detail_width)

    def __getattr__(self, item):
        if item in "xywh":
            if self.box is None:
                self.box = cv2.boundingRect(self.points)
            return self.box["xywh".index(item)]
        if item == "centre":
            return [self.x + self.w//2, self.y + self.h//2]
        if item == "length":
            if self.length is None:
                self.length=cv2.arcLength(self.points, False)
        return getattr(self, item)

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return f"Contour at {self.x, self.y}, size: {self.w,self.h}"