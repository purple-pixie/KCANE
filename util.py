import cv2
from PIL import Image
import pytesseract
import os
import random
def read_key(directions = True):
    key = 256
    while key > 127:
        key = cv2.waitKeyEx(0)
        if directions:
            print(key)
            if key in (2490368,
                        2621440,
                        2424832,
                        2555904,
                        ):  # up, down, left, right
                return key
    return key&0xFF
from pathlib import Path

def naked_filename(path):
    return (Path(path).resolve().stem)


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
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