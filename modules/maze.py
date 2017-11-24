from enum import Enum
import numpy as np
import robot_arm
from util import *


#======================#
#useful numbers#
boundaries = [([0,0,150], [50, 50, 255], "red"),
              ([30, 100, 30 ], [80, 255, 110], "green"),
              ([180, 180, 180], [255, 255, 255], "white")
              ]

im_width, im_height = 84, 84

buttons = [(74, 16), (135, 75), (74, 142), (10, 76)]

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

#======================#
#Enums#
class DIR(Enum):
    NONE = -1
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

#======================#
#Helper functions#
def image_to_maze_coords(x, y):
    return ((x)//14, (y)//14)
def maze_to_image_coords(x, y):
    return (x * 14 + 7, y * 14 + 7)

def hline(dst, x, y, color = (0,0,0)):
    cv2.line(dst, (14 * x, 14 * (y + 1)), (14 * (x + 1), 14 * (y + 1)), color, 2)
def vline(dst, x, y, color = (0,0,0)):
    cv2.line(dst, (14 * (x + 1), 14 * y), (14 * (x + 1), 14 * (y + 1)), color, 2)
#======================#
#Parent solver
class Solver():
    def __init__(self):
        self.mazes = np.load("data/modules/maze/mazedump.npy")
        #log# initialised
    def new(self, robot : robot_arm.RobotArm):
        return MazeSolver(robot, self.mazes)
    def identify(self, robot: robot_arm.RobotArm):
        image = robot.grab_selected()[36:36+im_height, 30:30+im_width]
        import itertools
        #there's bound to be an elegant way to do this without for loops
        #right now I can't think of it
        maze_id = -1
        marks = []
        for x, y in itertools.product(range(6), range(6)):
            im = image[14 * y:14 * (y + 1), 14 * x:14 * (x + 1)]
            green = np.sum(im[:,:,1]>150)
            if green > 25:
                marks.append((x, y))
                if len(marks) > 2:
                    return False
        if len(marks) != 2:
            return False
        #we found exactly two markers ... still, we should make sure they are in valid places
        if marks[0] not in markers or marks[1] not in markers:
            return False
        #they both were. Let's just make sure they both match the same maze
        return markers[marks[0]] == markers[marks[1]]

#======================#
#solver proper
class MazeSolver():
    def __init__(self, robot : robot_arm.RobotArm, mazes):
        self.robot = robot
        self.update_image()
        self.maze = -1
        self.start = None
        self.target = None

        for lower, upper, title in boundaries:
            image = self.image.copy()
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(image, lower, upper)
            output = cv2.bitwise_and(image, image, mask=mask)
            im=cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            cnts = contour(im)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            ##TODO: sanity check contours - if given a crap image this can divide by zero
            ## Or just give it a try/catch
            maze_id = -1
            marker_x, marker_y = (0,0)
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
                    #check that we aren't being told contradictory information
                    if maze_id > -1:
                        if markers[(x, y)] != maze_id:
                            print(f"Old marker {marker_x, marker_y} says maze id {maze_id}, new marker {x, y} is {markers[(x,y)]}!")
                    marker_x, marker_y = x, y
                    maze_id = markers[(x, y)]
                    self.maze = mazes[maze_id]

    def update_image(self):
        module = self.robot.grab_selected()
        self.image = module[36:36+im_height, 30:30+im_width]

    def solve(self):
        if self.start == None:
            #if already solved, we won't be able to find a start
            return
        maze = abstractmaze(self.maze, self.start, self.target)
        maze.draw(True)
        sol = maze.get_solution()
        self.robot.draw_module(maze.image)
        for dir in sol:
            x, y = buttons[dir.value]
            self.robot.moduleto(x, y)
            self.robot.click(0.2, 0.2)
        return True
#==================#
#abstract representation of a maze
# Shouldn't need any knowledge of where the mazes come from
# but it does, because I wanted drawing functions that look like KTANE mazes

class abstractmaze:
    def __init__(self, maze, start = (0,0), target = (5,5), image=None):
        self.image = np.full((im_height, im_width, 3), 130, dtype="uint8")
        self.already_tried = np.zeros((6, 6), dtype=np.bool_)
        self.start = start
        self.target = target
        if maze is -1:
            raise IndexError("Not given a maze, nor asked to make one")
        if maze is None:
            self.build_maze()
        else:
            self.hedge_mask=maze[:36].reshape((6, 6))
            self.vedge_mask=maze[36:].reshape((6, 6))

    def get_solution(self):
        """return a list of steps (as DIR items) to go from start to target"""
        win, path = self.solve_recursive(*self.start, *self.start)
        if win:
            #strip off the first 'move', because it doesn't have a direction / isn't a real move
            return path[1:]
        else:
            print("No path found")
            return []
        
    def draw_move(self, x, y, xx, yy, failed = False):
        color = (0,0,230) if failed else (20,230,20)
        cv2.line(self.image, maze_to_image_coords(x, y), maze_to_image_coords(xx, yy), color)

    def solve_recursive(self, x, y, prev_x, prev_y, direction = DIR.NONE):
        #recursively solve the maze
        if (x, y) == self.target:
            self.draw_move(x, y, prev_x, prev_y)
            return True, [direction]
        if self.already_tried[y][x]:
            return False, None
        self.already_tried[y][x] = True
        for xx, yy, dr in self.moves(x, y):
            win, path = self.solve_recursive(xx, yy, x, y, dr)
            if win:
                self.draw_move(x, y, prev_x, prev_y)
                return True, [direction] + path
        #draw failed moves
        #maybe if debug
        #self.draw_move(x, y, prev_x, prev_y, True)
        return False, None

    def moves(self, x, y):
        """generate the directions that we can move in from position x, y
        returns (x, y, direction [as a DIR enum])"""
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
        """legacy code - generate maze index 0 from manual data"""
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
        """Draw the maze onto a new image and return it
        if extras, draw the start and end points"""
        im = self.image
        for y in range(6):
            for x in range(6):
                #draw the dots
                cv2.rectangle(im, tuple(np.array(maze_to_image_coords(x, y))-1),
                              tuple(np.array(maze_to_image_coords(x, y))+1), (100,100,100), -1)

                #draw any walls present
                if self.hedge_mask[y][x]:
                    hline(im, x, y)
                if self.vedge_mask[y][x]:
                    vline(im, x, y)
        if extras:
            #draw the start / target
            x, y = self.target
            cv2.circle(im, tuple(maze_to_image_coords(x, y)), 8, (0, 0, 255), -1)
            x, y = self.start
            cv2.circle(im, tuple(maze_to_image_coords(x, y)), 5, (50, 50, 220), -1)

    ##quick and dirty wrappers to draw the horizontal and vertical lines on a maze image


    #TODO: make a save function?
    #TODO: make a generic version of this interface for future modules?
    def build_maze(self):
        self.hedge_mask = np.zeros((6,6), dtype=np.uint8)
        self.vedge_mask = np.zeros((6,6), dtype=np.uint8)
        im = self.draw()
        x, y = (0, 0)
        while True:
            temp = im.copy()
            vline(temp, x, y, (140, 140, 20))
            hline(temp, x, y, (140, 140, 20))
            char = read_and_display(temp, scale = 4)
            if char == ord("v"):
                if self.vedge_mask[y][x]:
                    vline(im, x, y, (255, 255, 255))
                    self.vedge_mask[y][x] = 0
                else:
                    vline(im, x, y)
                    self.vedge_mask[y][x] = 1
            elif char == ord("h"):
                if self.hedge_mask[y][x]:
                    hline(im, x, y, (255, 255, 255))
                    self.hedge_mask[y][x] = 0
                else:
                    hline(im, x, y)
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