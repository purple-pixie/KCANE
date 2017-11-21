#https://www.reddit.com/r/Python/comments/7cvtc5/whats_everyone_working_on_this_week/dq0i6mw/
#reply to this guy

from enum import Enum
import numpy as np
import screen
from robot_arm import RobotArm
from util import *
import modules.password
import modules.memory
import modules.maze

title_height = 25

def find_bomb(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur and canny
    blurred = cv2.bilateralFilter(gray, 21, 17, 17)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #display(blurred)
    edged = cv2.Canny(blurred, 30, 200)
    # display(edged)

    # find contours
    (im, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    def isfour(c):
        peri = cv2.arcLength(c, True)
        return cv2.approxPolyDP(c, 0.01 * peri, True) == 4

    #cnts = filter(isfour, cnts)
    def rect_area(x):
        x,y,w,h = cv2.boundingRect(x)
        return w * h
    cnts = sorted(cnts, key=rect_area, reverse=True)[:1]
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), random_color(), 2)
    for cnt in cnts:
        #cv2.drawContours(image, [cnt], -1, , 3)
        pass
    display(image)

def template(im, template, dst = None, threshold = 0.82):
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template, cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    if dst is not None:
        for pt in zip(*loc[::-1]):
            cv2.rectangle(dst, pt, (pt[0] + w, pt[1] + h), random_color(), 1)
    return loc


#s = screen.Screen(2)
#x=RobotArm(s)

#im = cv2.imread("screen/edges/0.bmp")
#for i in range(4):
    #glob.glob("screen/edges/*.bmp"):

im = cv2.imread(f"screen/{0}.bmp")
#cv2.rectangle(im, (0, 420), (126, 599), (0, 0, 0), -1)
#i2 = cv2.imread(f"screen/{8}.bmp")
#display(cv2.absdiff(im, i2))
#display(im)

def centre_cols(im):
    return im[...,250:550,...]
def centre_rows(im):
    return im[160:420]
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




#126,420 - clock

def get_corners(points):
    if len(points) == 0:
        return ()
    return sorted(points,key=lambda x:np.sum(x)-np.sum(points[0]))

def dist(a, b):
    return (a[0]-b[0]) ** 2 + (a[1] - b[1]) ** 2

def join(im1, im2):
    if im1.shape[-1] != im2.shape[-1]:
        print(f"Different channel sizes ({im1.shape[-1]} vs {im2.shape[-1]})")
        return im1


def read_edge(im):
    #im=s.grab()
    #im = cv2.imread("screen/11.bmp")
    if im.shape[-1] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    imcopy = im.copy()
    warped = imcopy
    #display(im)
    d = find_screws(im)
    if len(d) == 3:
        pass
    if len(d) == 4:
        c = get_corners(d)
        short = dist(c[1], c[0])
        long = dist(c[2], c[0])
        corners = np.float32([[150, 0], [50, 0], [150, 600], [50, 600]])
        pts = np.float32(c)
        M = cv2.getPerspectiveTransform(pts,corners)
        warped = cv2.warpPerspective(im,M,(200, 600))
        flipped = cv2.flip(warped,1)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        grayf = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
        _, threshf = cv2.threshold(grayf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #cv2.imwrite("warped.bmp", warped)
        #time.sleep(0.5)
        if False:
            w3 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            w2 = cv2.cvtColor(threshf, cv2.COLOR_GRAY2BGR)

            #im = np.concatenate((im, warped), axis=1)
            print(f"{im.shape}, {w3.shape}")
            im = np.concatenate((imcopy, w3), axis=1)
            im = np.concatenate((imcopy, w2), axis=1)
            #text = pytesseract.image_to_string(Image.open("warped.bmp"))
            #cv2.imshow("canny", w3)
            #roughly 80x500 for long edge
            display(imcopy)
        return warped
    return imcopy



def edge_realtime_test():
    s = screen.Screen(2)
    while "Displaying":
        im = read_edge(s.grab())
        if not keep_showing(im, lambda x: dump_image(x, "edgedumps/")):
            break

#edge_test()
def thresh_test():
    im = cv2.imread("edgedumps/9.bmp")
    gray = cv2.imread("edgedumps/9.bmp", 0)
    _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(_)
    im = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    temp = cv2.imread("templates/AA.bmp", 0)
    temp = cv2.resize(temp, (66, 178), interpolation=cv2.INTER_CUBIC)
    _,can = cv2.threshold(temp,_,255,cv2.THRESH_BINARY)
    #cv2.imwrite("resize.bmp", temp)
    display(can)
    template(im, temp, True, threshold=0.5)
    display(im)

#thresh_test()

def edge_test():
    s=screen.Screen(2)
    arm = RobotArm(s)
    edges = list(arm.examine_gubbins())
    for im in edges:
        edge = read_edge(im)
        #display(edge)
        dump_image(im, "screen/edges/")
        dump_image(edge, "edgedumps/")

def more_edges():
    im1 = cv2.imread("edgedumps/17.bmp")
    im2 = im1.copy()
    find_screws(im1, im2)
    display(im2)



class SOLVER(Enum):
    Password = 0
    Maze = 1
    Memory = 2

def init_solvers():
    solvers = {}
    solvers[SOLVER.Password] = modules.password.Password()
    solvers[SOLVER.Maze] = modules.maze.Maze()
    solvers[SOLVER.Memory] = modules.memory.Memory()
    return solvers

if __name__ == "__main__":
    ###TODO: detect modules automatically
    solvers = init_solvers()
    print(solvers)
    s = screen.Screen(2)
    r = RobotArm(s)
    pw = solvers[SOLVER.Password].new(r)
    pw.solve()

