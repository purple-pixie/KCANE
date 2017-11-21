import abc
import robot_arm

class Solver(abc.ABC):
    @abc.abstractmethod
    def solve(self):
        """solve the module. assumes it is already selected"""
        return
    @abc.abstractmethod
    def identify(self, image):
        """return a value from 0-1 of how much image looks like an instance of this module"""
        return 0