from enum import Enum
import numpy as np
import robot_arm
from util import *
from math import pi
from matplotlib import pyplot as plt
import screen
import logging
log = logging.getLogger(__name__)

class Solver():
    def new(self, robot):
        return SimpleWires(robot)

    def identify(self, image):
        return False


