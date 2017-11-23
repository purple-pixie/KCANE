from enum import Enum
import numpy as np
import robot_arm
from util import *
import itertools
from util import *
import logging
log = logging.getLogger(__name__)


def rect_area(x):
    x, y, w, h = cv2.boundingRect(x)
    return w * h
def rect_corners(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y), (x + w, y + h)

def find_highlight(image):
    """is there a highlighted module in image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([26,120,150], dtype="uint8")
    upper = np.array([30,255,255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    #mask should now have a very solid yellow square if there is one and very little noise
  #  print(np.sum(mask)//255)
  #  display(image, "hey", do_wait=False)
  #  display(mask, "dave")
    if np.sum(mask) > 2000 * 255:
        return True
        pass
    return False
    ##TODO: Test this with Morse flash on screen and with Simon flashing on screen
    ##TODO: determine the size of the highlight or actually return it
    #return False


def find_bomb_region(image):
    """get the boundingRect of the bomb"""
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    mask = np.logical_and(r > g, r > b)
    b[mask] = 0
    _, thresh = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im = image.copy()
    cnt = sorted(contour(thresh), key=rect_area, reverse=True)[0]
    rect = cv2.boundingRect(cnt)
    print(f"rect1 {rect}")
    p1, p2 = rect_corners(cnt)
    print(f"corners: {p1, p2}")
    cv2.drawContours(im, [cnt], 0, (0, 0, 255), 2)
    cv2.rectangle(im, p1, p2, (0,255,0), 4)
    display(im, "contours")#, do_wait=False)
    #display(thresh, "let")
    return rect