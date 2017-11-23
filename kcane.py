from enum import Enum
import numpy as np
import screen
from robot_arm import RobotArm
from util import *
from modules import solvers
import logging
logging.basicConfig(filename='KCANE.log', filemode='w', level=logging.DEBUG)

from util import *
import logging
log = logging.getLogger(__name__)


class Robot():
    def __init__(self, screen, window_name = None):
        self.screen = screen
        self.arm = RobotArm(screen, window_name = window_name)
        self.modules = [[None] * 6, [None] * 6]
        #front face. Back is 1
        self.face = 0
        #self.go()
    def go(self):
        self.analyse_bomb()
        self.defuse_bomb()

    def flip_bomb(self):
        self.arm.rotate(4)
        # we're now facing the other side
        self.face = 1-self.face

    def defuse_bomb(self):
        self.defuse_face()
        ##test for already winning?
        ##probably don't need to, trying to flip will take <1 second and it should find 0 modules after
        ## almost certianly don't need to. We examine front side then back, so we solve back first.
        ## front will never(?) not have modules on it
        self.flip_bomb()
        sleep(0.5)
        self.defuse_face()

    def defuse_face(self):
        for mod in range(6):
            if self.modules[self.face][mod]:
                self.arm.goto(mod)
                sleep(0.2)
                module = self.identify()
                if module is None:
                    print("Not sure what this is. Skipping")
                    #dump_image(self.arm.grab_selected())
                else:
                    print(f"Looks like {module}. Defusing")
                    solvers[module].new(self.arm).solve()
                self.arm.rclick(after=0.25)

    def examine_gubbins(self):
        pass

    def analyse_bomb(self):
        #reset ourselves to a picked up, centred bomb with no module selected
        self.arm.wake_up()
        self.modules[self.face] = list(self.arm.scan_modules())
        self.flip_bomb()
        self.modules[self.face] = list(self.arm.scan_modules())
        #print(f"active modules: {self.modules}")
        
    
    def identify(self):
        im = self.arm.grab()
        static_screen = screen.Screen(image=im)
        fake_arm = RobotArm(static_screen)
        for module in solvers:
            if solvers[module].identify(fake_arm):
                return module
        

def main():
    #s = screen.Screen(image=cv2.imread("dump/maze/0.bmp"))
    s = screen.Screen(2)
    r = Robot(s, "thoughts")
    #r.arm.goto(0,0.2)
#    p = solvers["maze"].identify(r.arm)
    r.go()
    #p.solve()
    #r.panic("test/")

from bomb_examiner import *

if __name__ == "__main__":
#   s=screen.Screen(2)
#   import time
#   for i in range(10):
#       s.save_screen("test/")
#       time.sleep(1)

 #  ims = images_in("test/")
 #  for im in ims:
 #      find_highlight(im)
    main()
