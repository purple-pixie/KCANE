import numpy as np
import robot_arm
import itertools
from util import *
import logging
log = logging.getLogger(__name__)
from operator import itemgetter

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



#================================================
#Junk copied from Scratch

import cv2

def template(im, template, dst = None, threshold = 0.82):
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template, cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    if dst is not None:
        for pt in zip(*loc[::-1]):
            cv2.rectangle(dst, pt, (pt[0] + w, pt[1] + h), random_color(), 1)
    return loc

screw = cv2.imread("templates/screw.bmp", 0)

def find_screws(im, dst = None):
    #for im in glob.glob("screen/*.bmp"):
    x=template(im, screw)
    pts = np.array(sorted(zip(*x[::-1]), key=np.product ))
    if len(pts) == 0:
        return [], []
    pts = pts + (len(screw)//2, len(screw[0])//2)
    last = pts[0]
    out = [last]
    for i in range(len(pts)-1):
        current = pts[i+1]
        diff = np.sum(current) - np.sum(last)
        if np.sum(cv2.absdiff(current, last)) < 5:
            continue
        else:
            out.append(current)
            last = current
    #print(f"definite: {definite}, \n maybe: {maybe}")
    h,w = screw.shape
    if dst is not None:
        for d in out:
            cv2.rectangle(dst, tuple(d), tuple(d+screw.shape), (0, 255, 0), 2)
    return out


def get_corners(points):
    s_points = sorted(points, key=np.sum)
    orig = s_points[0]
    return sorted(points, key=lambda x: dist(x, orig))

def dist(a, b):
    return (a[0]-b[0]) ** 2 + (a[1] - b[1]) ** 2

def test_edges():
    for name, image in images_in("dump/edges/", return_names=True):
       # image = get_edge(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #image = cv2.cvtColor(hsv , cv2.COLOR_HSV2BGR)
        #count_batteries(image, hsv)
        find_serial(image, hsv)

def find_indicators(hsv):
    #todo find indicators
    # very well defined circles with very high value, low hue/sat
    # grab the region to the right of them
    # clean trim and tesseract it
    return []

def tess_region_from_contour(image, cnt, config = "--psm 10 digits"):
    x, y, w, h = cv2.boundingRect(cnt)
    print(f"W: {w}, H: {h}")
    im = image[y-2:y+h+2,x-1:x+w+1]
    display(im)
    return tess(im, config)

def get_serial_letters(image):
    cnts = contour(image)
    im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for cnt in cnts:
        cv2.drawContours(im, [cnt], 0, (0,0,255), 1)
    display(im)
    #first sort by size to get the 6 characters
    if len(cnts) < 6:
        #didn't find digits
        print(f"Failed {len(cnts)}")
        return
    boxes = [(cv2.boundingRect(i), i) for i in cnts]
    by_area = sorted(boxes, key = lambda x: x[0][2] * x[0][3], reverse=True)[:6]
    by_x = sorted(by_area, key=lambda x: x[0][0])
    print(by_x[0][0])
    #regions = sorted(cnts, key=cv2.contourArea, reverse=True)[:6]
    #then sort by x-value to get them in order
    #regions = sorted(regions, key = lambda x: cv2.boundingRect(x)[0])
    for box,cnt in by_x[:-1]:
        text= tess_region_from_contour(image, cnt, "--psm digitletters")
        #print(f"tessL {text}")
    digit = by_x[-1][1]
    text = tess_region_from_contour(image, digit, "--psm 10 digits")
    #print(f"tessD {text}")



def find_serial(bgr, hsv):
    canvas = bgr.copy()
    lower = np.array([10,30,180], dtype="uint8")
    upper = np.array([30,50,225], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    blur = cv2.GaussianBlur(mask,(11,11),0)
    #display(blur)
    #get_serial_letters(blur)
    #display(np.hstack((mask, blur)))
    #TODO: find serial number
    #get contours
    #get bounding rects
    #find something roughly serial sized
    #clean, trim and tesseract it
    contours = contour(blur)
    #display(blur)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 120 < w < 150:
            region = hsv[y:y+h,x:x+w]
            #display(region)
            mask = inRange(region, [2,0,0], [50,50,100])
            #display(mask)
            #get_serial_letters(mask)
            text = tess(mask, config="--psm 8 letters", remove=False)
            digits = tess(mask, config="--psm 8 digits", remove=False)
            dump_image(mask, dir="serial")
            print(f"Tess: {text} | {digits}")

            #TODO: Do this properly and be less of a hack
            #interpret the last letter as a digit and the rest as letters
            #will probably mistake 1s for Is but otherwise should find vowels accurately
            #in spite of tess's awful accuracy
            return text[:-1]+digits[-1]
            #cv2.drawContours(canvas, [cnt], -1, random_color() ,1)
           # display(mask)
    return ""

def find_ports(hsv):
    lower = np.array([160,100,140], dtype="uint8")
    upper = np.array([170,255,255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    #display(mask)
    #mask is just the pink pixels
    #if there are any it should be a parallel port
    #and that's all we care about (?)
    return np.sum(mask) > 10

def count_batteries(image, hsv, draw = True):
    #for name, image in images_in("dump/gubbins/", return_names=True):
    #image = cv2.imread("dump/gubbins/A.bmp")
    #can = np.full(image.shape, 255, dtype="uint8")
    canvas = None
    if draw:
        canvas = image.copy()
    lower = np.array([16,180,120], dtype="uint8")
    upper = np.array([25,255,255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    cnts = contour(mask)
    batteries = []
    for cnt in cnts:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if radius < 8:
            continue
        if draw:
            cv2.circle(canvas, center, radius, (0, 255, 0), 2)

        for c, r in batteries:
            if dist(c, center) < max(radius, r) ** 2:
                if r < radius:
                    batteries.remove((c,r))
                    batteries.append((center, radius))
                    if draw:
                        cv2.circle(canvas, c, r, (0, 0, 255), 2)
                    break
                else:
                    if draw:
                        cv2.circle(canvas, center, radius, (0, 0, 255), 2)
                    break
        else:
            batteries.append((center, radius))

    if len(batteries) % 2:
        print(f"Only found {len(batteries)} battery ends, that seems wrong")
        dump_image(canvas, dir="fail", starts="battery")
        dump_image(image, dir="fail", starts="battery")

    #display(canvas)
    return int(np.floor(len(batteries)/2))
    #display(can)

 #   h, s, v = (hsv[...,i] for i in range(3))
 #   h *= 2
 #   v = cv2.subtract(v, s)
 #   im = np.hstack(((h,s,v)))
 #   display(im)
 #   cv2.destroyAllWindows()

def get_edge(im, dir = 0):
    h, w = im.shape[:2]
    imcopy = im.copy()
  #if dir == 0:
  #    M = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1)
  #    w, h = h, w
  #imcopy = cv2.warpAffine(im, M, (h, w))
    warped = imcopy
    d = find_screws(im)
    if len(d) < 3:
        #no way we can work with this. abort
        print(f"Failed to find enough screws ({len(d)})")
        return None
    if len(d) == 3:
        #invent a 4th point?
        #should be very doable given exactly 3
        pt1, pt2, pt3 = d
        if (pt1[0] - pt2[0]) < 10:
            x = pt3[0]
            if abs(pt3[1] - pt1[1]) < 10:
                y = pt2[1]
            else:
                y = pt1[1]
        else:
            y = pt3[1]
            if abs(pt3[0] - pt1[0]) < 10:
                x = pt2[0]
            else:
                x = pt1[0]
        d += [(x, y)]
    elif len(d) > 4:
        #group possible points into combinations of 4
        #sort by how rectangular they are
        #set list of points to just be the 4 that make the best rectangle
        d = np.array(sorted(itertools.combinations(d, 4), key=rectness)[0])
    c = get_corners(d)
    short = dist(c[1], c[0])
    long = dist(c[2], c[0])
    corners = np.float32([[0, 150], [0, 50], [600, 150] , [600, 50]])
    pts = np.float32(c)
    M = cv2.getPerspectiveTransform(pts,corners)
    warped = cv2.warpPerspective(im,M,(600, 200))
    if dir == 0:
        #rotate it 180
        return cv2.resize(cv2.flip(warped, -1), None, fx=2/3, fy=1)
        ##WORKS
    if dir == 1:
        # flip vertical
        return cv2.flip(warped, 0)
    if dir == 2:
        #
        return cv2.resize(warped, None, fx=2/3, fy=1)
    if dir == 3:
        return cv2.flip(warped, 1)

def rectness(points):
    r = np.array(points)
    xs = r[:, 0]
    ys = r[:, 1]
    xx = sorted(xs)
    yy = sorted(ys)
    xdiff = abs(xx[1] - xx[0]) + abs(xx[3] - xx[2])
    ydiff = abs(yy[1] - yy[0]) + abs(yy[3] - yy[2])
    return xdiff + ydiff