import abc
import robot_arm

class Solver(abc.ABC):
    """solves one instance of the owned module"""
    @abc.abstractmethod
    def __init__(self, robot:robot_arm.RobotArm):
        return
    @abc.abstractmethod
    def solve(self):
        """solve the module. assumes it is already selected"""
        return
    def update_image(self):
        self.image = self.robot.grab_selected()

class SolverModule(abc.ABC):
    """oversees all the solving of instances of the module class"""

    @abc.abstractmethod
    def new(self, robot:robot_arm.RobotArm)-> Solver:
        """return a solver for a new instance of this module
        REQUIRES that the specified module be currently selected in-game
        Things will break in ugly ways otherwise"""
        return Solver()
    @abc.abstractmethod
    def identify(self, image):
        """return a value from 0-1 of how much image looks like an instance of this module
        image should be a module-sized 3-channel image from robot_arm.RobotArm.grab_selected(True) """
        return 0
