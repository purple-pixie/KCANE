import cv2
import numpy as np
from util import *
import itertools
#==============#
#numbers#
xpad, ypad = 20, 30
xstep, ystep = 20, 20
width, height = 100, 100



class BombDrawer:
    def __init__(self, robot):
        self.image = np.full((600,800,3), 200, dtype="uint8")
        self.robot = robot
        self.window_title = "Defuser"
        cv2.namedWindow(self.window_title)
        self.draw()
    def show(self):
        cv2.imshow(self.window_title, self.image)
        cv2.waitKey(1)
    def draw_module(self, image, pos, nonwhite = False):
        face_region = get_face_region(self.image, self.robot.face)
        region = get_module_region(face_region, pos)

        if not image.shape == region.shape:
            image = cv2.resize(image, region.shape[:2], interpolation=cv2.INTER_AREA)
            cv2.rectangle(image,(0,0),image.shape[:2], (0,0,0))
        if nonwhite:
            mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 255
            region[mask] = image[mask]
        else:
            np.copyto(region, image)
        self.show()

    def draw(self):
        for face in (0,1):
            image = get_face_region(self.image, face)
            for mod in range(6):
                region = get_module_region(image, mod)
                label = self.robot.modules[face][mod]
                color = (130,130, 130)
                if label == "unknown":
                    color = (200,50,220)
                    cv2.putText(region, "?", (15, 90), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (255,55,200), 4)
                elif label != "empty":
                    #cv2.rectangle(region, (0,0), (width, height), color, -1)
                    color = (20, 240, 240)
                else:
                    cv2.rectangle(region, (0,0), (width, height), color, -1)
                cv2.rectangle(region, (0,0), (width, height), color, 4)

            #region *
            #display(region)
        self.show()

face_height = ypad + 2 * (ystep + height)
face_width = 2 * xpad +  3 * (xstep + width)
def get_face_region(image, face):
    y1 = face * (face_height)
    y2 = y1 + face_height
    return image[y1:y2, : face_width]
def get_module_region(face_image, pos):
    row = pos // 3
    col = pos % 3
    x1 = xpad + (xstep + width) * col
    y1 = ypad + (ystep + height) * row
    x2, y2 = x1 + width, y1 + height
    return face_image[y1:y2, x1:x2]

    #def