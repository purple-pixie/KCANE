from enum import Enum
import numpy as np
import screen
from robot_arm import RobotArm, sleep
from util import *
from modules import solvers
from bomb_drawer import BombDrawer
import logging
logging.basicConfig(filename='KCANE.log', filemode='w', level=logging.DEBUG)
log = logging.getLogger(__name__)

module_images = dict(images_in("images/", return_names=True))

class Robot():
    def __init__(self, screen):
        self.screen = screen
        self.arm = RobotArm(screen, robot=self)
        self.modules = [["unknown"] * 6, ["unknown"] * 6]
        #front face. Back is 1
        self.drawer = BombDrawer(self)
        self.face = 0
        self.selected = 0
        self.battery_count = 0
        self.gubbins = {}
        self.serial_odd = -1
        self.serial_vowel = -1
    def peek(self):
        self.examine_gubbins()
        self.analyse_bomb()


    def flip_bomb(self):
        self.arm.rotate(4)
        # we're now facing the other side
        self.face = 1-self.face
        #make sure we caught up
        #sleep(0.25)
    def defuse_bomb(self):
        self.defuse_face()
        ##test for already winning?
        ##probably don't need to, trying to flip will take <1 second and it should find 0 modules after
        ## almost certianly don't need to. We examine front side then back, so we solve back first.
        ## front will never(?) not have modules on it
        self.flip_bomb()
        self.defuse_face()
        self.flip_bomb()

    def defuse_face(self):
        for pos in range(6):
            if self.modules[self.face][pos] != "empty":
                self.selected = pos
                self.arm.goto(pos, after = 0.5)
                module = self.identify()
                if module is None:
                    print("Not sure what this is. Skipping")
                    self.draw_module(module_images["unknown"], nonwhite=True)
                    #dump_image(self.arm.grab_selected())
                else:
                    if module == "solved":
                        print(f"Already solved, ignoring")
                    else:
                        print(f"Looks like {module}. Defusing")
                        solved = solvers[module].new(self.arm).solve()
                        if solved:
                            print(f"Solved it")
                            self.draw_module(module_images["solved"], nonwhite=True)
                        else:
                            print(f"Failed")

                self.arm.rclick(after=0.45)

    #185, 303: 104x54


    def examine_gubbins(self):
        self.arm.wake_up()
        self.arm.mouse_to_centre()
        batteries = 0
        indicators = []
        parallel = False
        serial = ""
        for dir, edge in self.arm.get_edges():
            warped = get_edge(edge, dir)
            if not warped is None:
                #dump_image(warped, dir="edges")
                hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                if batteries < 3:
                    bats = count_batteries(warped, hsv)
                    if bats > 0:
                        batteries += bats
                        print(f"Found {bats} more batteries")
                    #dump_image(warped, dir="gubbins", starts=f"Bats{bats}_")

                if not parallel:
                    parallel = find_ports(hsv)
                    if parallel:
                        print("Found a parallel port")

                if serial == "":
                    serial = find_serial(warped, hsv)
                    if serial != "":
                        print(f"Found serial: {serial}")
                        self.serial_vowel = len(set("aeiou")&set(serial))
                        try:
                            self.serial_odd = int(serial[-1])%2
                        except ValueError:
                            log.error(f"could not convert {serial[-1]} to int for parity check")
                            self.serial_odd = 1
                            pass

                indicators += find_indicators(warped)

        for i in "batteries", "parallel", "serial", "indicators":
            self.gubbins[i] = eval(i)
        print(f"Gubbins results: {self.gubbins}")

    def serial_is_odd(self):
        log.info("checking oddity")
        if self.serial_odd == -1:
            log.info("checking oddity in detail")
            #go check the serial number
            #but need to make sure we dont break state first, mouse position / selected module et c
            #self.examine_gubbins()
            return 1
            pass
        return self.serial_odd
    def serial_has_vowel(self):
        if self.serial_vowel == -1:
            #read serial
            pass
        return self.serial_vowel
    def draw(self):
        self.drawer.draw()

    def draw_module(self, image, nonwhite = False):
        self.drawer.draw_module(image, self.selected, nonwhite=nonwhite)

    def analyse_face(self):
        sleep(0.1)
        for i, label in enumerate(self.arm.scan_modules()):
            self.modules[self.face][i] = label
            self.draw()
    def analyse_bomb(self):
        #reset ourselves to a picked up, centred bomb with no module selected
        self.arm.wake_up()
        self.analyse_face()
        self.flip_bomb()
        self.analyse_face()
        #print(f"active modules: {self.modules}")


    def identify(self, do_show = False):
        im = self.arm.grab()
        static_screen = screen.Screen(image=im)
        fake_arm = RobotArm(static_screen)
        #first check if it's already solved
        indicator = to_hsv(fake_arm.grab_selected()[:23 , 130:])
        green = inRangePairs(indicator, [(58, 74), (222, 255), (204, 251)])
       # display(indicator)
       # print(np.sum(green))
       # display(green)
        if np.sum(green) > 2550:
            #at least 10 pixels in the top right matched our (very) bright green mask
            #that's solved
            return "solved"
       #probably have enough dumps of centred modules now
       #dump_image(im, "fail", starts="identify")
        for module in solvers:
            id, img = solvers[module].identify(fake_arm)
            if do_show:
                display(img, module)
            if id:
                return module


    def watch(self):
        im = self.arm.grab_selected()
        ims = [im]
        while 1:
            im = self.arm.grab_selected()
            for base in ims:
                if np.sum(cv2.absdiff(im, base)) < 6000:
                    break
            else:
                ims.append(im)
                dump_image(im,"watched")

def main():
    s = screen.Screen(2)
    s = screen.Screen(image_path="img2.bmp")
    r = Robot(s)
    r.identify(True)
    #todo; Draw stuff
    #we like drawing
    #r.arm.goto(0,0.2)
    #p = solvers["simple_wires"]
    #print(p.identify(r.arm))
    #p.new(r.arm).solve()
    #cv2.waitKey(0)
    #return
    #r.watch()
    try:
        r.peek()
        r.defuse_bomb()
    except:
        #stops random crashes leaving right click clicked
        #mouse was inited before the try so it shouldn't be causing IO errors during the except
        robot_arm.mouse.release(robot_arm.Button.right)
        raise
    #p.solve()
    #r.panic("test/")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

from bomb_examiner import *

if __name__ == "__main__":
    main()