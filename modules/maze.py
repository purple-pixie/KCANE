import glob
from enum import Enum
import itertools
import numpy as np
import pytesseract
import robot_arm
from modules.solver import Solver
from util import *

boundaries = [([0,0,150], [50, 50, 255], "red"),
              ([30, 100, 30 ], [80, 255, 110], "green"), #green
              ([180, 180, 180], [255, 255, 255], "white") #white
              ]

im_width = 115-30
im_height = 120-36

buttons = [(74, 16), (135, 75), (74, 142), (10, 76)]

def image_to_maze_coords(x, y):
    return np.array(((x)//14, (y)//14))
def maze_to_image_coords(x, y):
    return np.array((x * 14 + 7, y * 14 + 7))

#mapping from marker-coordinates to maze index
markers = {(0, 1): 0, (5, 2): 0,
           (4, 1): 1, (1, 3): 1,
           (3, 3): 2, (5, 3): 2,
           (0 ,0): 3, (0, 3): 3,
           (4, 2): 4, (3, 5): 4,
           (2, 4): 5, (4, 0): 5,
           (1, 0): 6, (1, 5): 6,
           (3, 0): 7, (2, 3): 7,
           (2, 1): 8, (0, 4): 8
           }

class Maze(Solver):
    def __init__(self, robot : robot_arm.RobotArm):
        self.robot = robot
        self.update_image()
        self.maze = None
        self.start = None
        cv2.imwrite("maze.bmp", self.image)
        #display(self.image)
        print(self.image.shape)
        if self.image.shape[2] == 4:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)
        for lower, upper, title in boundaries:
            image = self.image.copy()
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(image, lower, upper)
            output = cv2.bitwise_and(image, image, mask=mask)
            im=cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            cnts = contour(im)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            #print(len(cnts))
            for cnt in cnts:
                cv2.drawContours(im, [cnt], -1, random_color(), 2)
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                x, y = image_to_maze_coords(cx, cy)
                if title == "red":
                    self.target = (x, y)
                elif title == "white":
                    self.start = (x, y)
                else:
                    self.maze = markers[(x, y)]
                #print(f"{cx, cy} -> maze co-ords: {image_to_maze_coords(cx, cy)}")
            # show the images
            #display(im, title)
#        print(f"maze id {self.maze}, pos {self.start}, target {self.target}")

    def update_image(self):
        module = self.robot.grab_selected()
        cv2.imwrite("module.bmp", module)
        self.image = module[36:36+im_height, 30:30+im_width]

    def solve(self):
        maze = abstractmaze(self.maze, self.start, self.target)
        sol = list(maze.solve())
        print(sol)
        for step in sol:
            print((step, buttons[step.value]))
            x, y = buttons[step.value]
            self.robot.moduleto(x, y)
            self.robot.click(0.2, 0.2)
        #print(sol)
    def identify(self, image):
        return False

class DIR(Enum):
    NONE = -1
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

#mazes = [np.frombuffer(b, dtype='|S1') == b'1' for b in [b'010011001110010010011110010010000000001000101000101000100100001010010100',
#         b'101001010110001010010100000010000000001000010100101000010110111010101000',
#         b'010000100110000000000000011000000000001100111010011010111110101110000100',
#         b'001110000110011010011110011100000000010000110000101010100000000010001010',
#         b'111100011011001100011010001110000000000000000010010100100110100010100000',
#         b'000100000010011001100000011010000000101000111010011100010110011100000100',
#         b'011000001110110101000110011100000000000100101010010100010010110010000000',
#         b'001000011110001100010111001111000000100100001010100010101000110000000000',
#         b'001100000100011010000110000001000000100000110110001010110100111010010100']]
#x = np.array(mazes)
#np.save("data/modules/maze/mazedump", x)


class abstractmaze:
    def __init__(self, id = None, start = (0,0), target = (5,5)):
        self.already_tried = np.zeros((6, 6), dtype=np.bool_)
        self.start = start
        self.target = target
        if id is None:
            im = self.draw()
            self.build_maze(im)
        else:
            #self.create_maze(id)
            self.load_maze(id)
        #display(self.draw())

    def load_maze(self, id):
        mazes = np.load("data/modules/maze/mazedump.npy")
        self.maze = mazes[id]
        self.hedge_mask=self.maze[:36].reshape((6, 6))
        self.vedge_mask=self.maze[36:].reshape((6, 6))

    def solve(self):

        win, path = self.solve_recursive(*self.start)
        if win:
            return path[1:]
            #print(f"Winning path: {list(reversed(path))}")
        else:
            print("No path found")
    def solve_recursive(self, x, y, direction = DIR.NONE):
        if (x, y) == self.target:
            return True, [direction]
        if self.already_tried[y][x]:
            return False, None
        self.already_tried[y][x] = True
        for xx, yy, dr in self.moves(x, y):
            win, path = self.solve_recursive(xx, yy, dr)
            if win:
                return True, [direction] + path
        return False, None
    def moves(self, x, y):
        if x > 0:
            if not self.vedge_mask[y][x-1]:
                yield x-1, y, DIR.LEFT
        if x < 5:
            if not self.vedge_mask[y][x]:
                yield x + 1, y, DIR.RIGHT
        if y > 0:
            if not self.hedge_mask[y-1][x]:
                yield x, y - 1, DIR.UP
        if y < 5:
            if not self.hedge_mask[y][x]:
                yield x, y + 1, DIR.DOWN
    def create_maze(self):
        hedges = [(1, 0), (4, 0), (5, 0),
                  (2, 1), (3, 1), (4, 1),
                  (1, 2), (4, 2),
                  (1, 3), (2, 3), (3, 3), (4, 3),
                  (1, 4), (4, 4)
                  ]
        vedges = [(2, 0), (2, 1), (2, 2),
                  (0, 1), (0, 2), (0, 3),
                  (3, 3), (2, 4), (4, 4),
                  (1, 5), (3, 5),
                  ]
        self.hedge_mask = np.zeros((6,6), dtype=np.uint8)
        self.vedge_mask = np.zeros((6,6), dtype=np.uint8)
        for x, y in hedges:
            self.hedge_mask[y][x] = 1
        for x, y in vedges:
            self.vedge_mask[y][x] = 1

    def draw(self, extras=False):
        im = np.ones((120-36, 115-30, 3), dtype = np.uint8)*255
        for y in range(6):
            for x in range(6):
                cv2.rectangle(im, tuple(maze_to_image_coords(x, y)-1), tuple(maze_to_image_coords(x, y)+1), (100,100,100), -1)
                if self.hedge_mask[y][x]:
                    cv2.line(im, (14*x, 14*(y+1)), (14*(x+1), 14*(y+1)), (00,0,0), 2)

                if self.vedge_mask[y][x]:
                    cv2.line(im, (14 * (x+1), 14 * y), (14*(x+1), 14 * (y+1)), (0, 0, 0), 2)
        if extras:
            x, y = self.target
            cv2.circle(im, tuple(maze_to_image_coords(x, y)), 8, (0, 0, 255), -1)
            x, y = self.start
            cv2.circle(im, tuple(maze_to_image_coords(x, y)), 5, (220, 220, 220), -1)
        return im
    def dump(self, dumpfile):
        with open(dumpfile, "w") as fil:
            for i in itertools.chain(self.hedge_mask.ravel(), self.vedge_mask.ravel()):
                fil.write(str(i))

    def reset(self):
        self.hedge_mask = np.zeros((6,6), dtype=np.uint8)
        self.vedge_mask = np.zeros((6,6), dtype=np.uint8)
    def hline(self, dst, x, y, color = (0,0,0)):
        cv2.line(dst, (14 * x, 14 * (y + 1)), (14 * (x + 1), 14 * (y + 1)), color, 2)
    def vline(self, dst, x, y, color = (0,0,0)):
        cv2.line(dst, (14 * (x + 1), 14 * y), (14 * (x + 1), 14 * (y + 1)), color, 2)
    def build_maze(self, im):
        self.reset()
        x, y = (0, 0)
        while True:
            temp = im.copy()
            self.vline(temp, x, y, (140, 140, 20))
            self.hline(temp, x, y, (140, 140, 20))
            char = read_and_display(temp, scale = 4)
            if char == ord("v"):
                if self.vedge_mask[y][x]:
                    self.vline(im, x, y, (255, 255, 255))
                    self.vedge_mask[y][x] = 0
                else:
                    self.vline(im, x, y)
                    self.vedge_mask[y][x] = 1
            elif char == ord("h"):
                if self.hedge_mask[y][x]:
                    self.hline(im, x, y, (255, 255, 255))
                    self.hedge_mask[y][x] = 0
                else:
                    self.hline(im, x, y)
                    self.hedge_mask[y][x] = 1
            #65362, 65364, 65361, 65363):# up, down, left, right
            if char == 2490368:
                y -= 1
            elif char == 2621440:
                y += 1
            elif char == 2424832:
                x -= 1
            elif char == 2555904:
                x += 1
            if char == ord(" "):
                break
        dir = "data/modules/maze/"
        filename = f"{dir}{len(glob.glob(f'{dir}*.maz'))}.maz"
        self.dump(filename)