from enum import Enum
import numpy as np
import screen
from robot_arm import RobotArm
from util import *
import modules.password
import modules.memory
import modules.maze
import modules.wire_sequence
import logging
logging.basicConfig(filename='KCANE.log', filemode='w', level=logging.DEBUG)

from modules.solver import Solver, SolverModule
from util import *
import logging
log = logging.getLogger(__name__)


class SOLVER(Enum):
    Password = 0
    Maze = 1
    Memory = 2
    Sequence = 3

def init_solvers():
    solvers = {}
    solvers[SOLVER.Password] = modules.password.Password()
    solvers[SOLVER.Maze] = modules.maze.Maze()
    solvers[SOLVER.Memory] = modules.memory.Memory()
    solvers[SOLVER.Sequence] = modules.wire_sequence.Sequence()
    return solvers

def main():
    solvers = init_solvers()
    s = screen.Screen(image=cv2.imread("lcds/img13.bmp", 0))
    s = screen.Screen(2)
    r = RobotArm(s)
    p = solvers[SOLVER.Password].new(r)
    #p.solve()
    #r.panic("test/")

from bomb_examiner import find_bomb

if __name__ == "__main__":
    ims = images_in("test/")
    for im in ims:
        find_bomb(im)

    # main()
