import numpy as np
import screen
from robot_arm import RobotArm, sleep
from util import *
from modules import solvers
from bomb_drawer import BombDrawer
import logging
import clock
logging.basicConfig(filename='KCANE.log', filemode='w', level=logging.DEBUG) #, format='%(asctime)s.%(msecs)03d %(message)s')
log = logging.getLogger(__name__)
log.info("started")
module_images = dict(images_in("images/", return_names=True))

class Robot():
    def __init__(self, screen, safe=False):
        self.safe = safe
        self.screen = screen
        self.arm = RobotArm(screen, robot=self)
        self.modules = [["unknown"] * 6, ["unknown"] * 6]
        #front face. Back is 1
        self.drawer = BombDrawer(self)
        self.face = 0
        self.selected = 0
        self.battery_count = 0
        self.gubbins = {}
        self.serial_odd = False
        self.serial_vowel = False
        self.strikes = 0

    def peek(self):
        self.examine_gubbins()
        self.analyse_bomb()

    def has_indicator(self, label):
        return False
        #label == "CAR"

    def flip_bomb(self):
        self.arm.rotate(4)
        # we're now facing the other side
        self.face = 1-self.face
        #make sure we caught up
        #rotate does this
        #sleep(1)

    def defuse_bomb(self):
        self.defuse_face()
        ##test for already winning?
        self.flip_bomb()
        self.defuse_face()
        self.flip_bomb()

    def defuse_face(self):
        self.selected = -1
        active = [pos for pos, mod in enumerate(self.modules[self.face]) if mod not in ("empty", "clock")]
        for pos in active:
            if self.selected == -1:
               self.arm.goto(pos)
            else:
               self.arm.goto_from(pos, after = 0.5)
            self.selected = pos

            #CLOCK STUFF DEBUG#
            clock_pos = self.get_clock_pos()
            if clock_pos != -1:
                c_x = clock_pos % 3
                x = self.selected % 3
                try:
                    im = self.arm.grab_other_module(clock_pos)
                    print(clock.read_clock(im))
                    display(im, "clock", wait_forever=False)
                except cv2.error:
                    print(f"failed to read clock {c_x} from {x}")

            #ENDCLOCK

            #dump_image(self.arm.grab(), dir="ident", starts="full")
            module = self.identify()
            if module is None:
                print("Not sure what this is. retrying ")
                sleep(0.2)
                module = self.identify()
            if module is None:
                print("Still not sure what this is. Skipping")
                self.draw_module(module_images["unknown"], nonwhite=True)
                #dump_image(self.arm.grab_selected())
            else:
                if module == "solved":
                    print(f"Already solved, ignoring")
                else:
                    print(f"Looks like {module}. Defusing")
                    solved = solvers[module].new(self.arm).solve()
                    state = self.arm.indicator_state()
                    if state == 0:
                        #if state initially gray, try sleeping for a bit and testing again
                        sleep(0.1)
                        state = self.arm.indicator_state()
                    if state == -1:
                        log.warning(f"Struck out on {module}!")
                        self.strikes += 1
                        self.draw_module(module_images["failed"], nonwhite=True)
                    if state == 0:
                        log.info(f"Module {module} says solved but module is unfinished")
                    if state == 1:
                        log.info(f"Solved {module}")
                        self.draw_module(module_images["solved"], nonwhite=True)
        if self.selected != -1: self.arm.rclick(after=0.35)

    def get_strikes(self):
        return self.strikes

    def __getitem__(self, item, default=None):
        return self.gubbins.get(item, default=default)


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
              # dump_image(warped, dir="edges", starts=f"{dir}_")
              # display(warped, f"edge {dir}", wait_forever=False)
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
                        self.serial_vowel = len(set("AEIOU")&set(serial)) > 0
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

    def serial_digit_error(self, made_error = False):
        self.serial_odd = not self.serial_odd
        self.strikes += made_error

    def serial_vowel_error(self):
        self.serial_vowel = not self.serial_vowel
        self.strikes += 1

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
            print("it really shouldnt be")
            self.serial_vowel = 0
        return self.serial_vowel
    def draw(self):
        self.drawer.draw()

    def draw_module(self, image, nonwhite = False):
        self.drawer.draw_module(image, self.selected, nonwhite=nonwhite)

    def analyse_face(self):
        sleep(0.5)
        for i, label in enumerate(self.arm.scan_modules(True)):
            self.modules[self.face][i] = label
            self.draw()
    def analyse_bomb(self):
        #reset ourselves to a picked up, centred bomb with no module selected
        self.arm.wake_up()
        sleep(0.5)
        self.analyse_face()
        self.flip_bomb()
        self.arm.wake_up()
        sleep(0.5)
        self.analyse_face()
        self.flip_bomb()
        #print(f"active modules: {self.modules}")

    def get_clock_pos(self):
        try:
            return self.modules[self.face].index("clock")
        except ValueError:
            log.debug(f"clock not found on face {self.face} | modules: {self.modules[self.face]}")
            return -1

    def identify(self, do_show=False):
        claims = sorted(self.get_identity_claims(do_show))
        if len(claims):
            return claims[0][1]


    def get_identity_claims(self, do_show = False):
        im = self.arm.grab()
        static_screen = screen.Screen(image=im)
        fake_arm = RobotArm(static_screen, self)
        #first check if it's already solved
        indicator = to_hsv(fake_arm.grab_selected()[:23 , 130:])
        green = inRangePairs(indicator, [(58, 74), (222, 255), (204, 251)])
       # display(indicator)
       # print(np.sum(green))
       # display(green)
        if np.sum(green) > 2550:
            #at least 10 pixels in the top right matched our (very) bright green mask
            #that's solved
            #TODO: split ID and solved-flag
            return "solved"
       #dump_image(im, "fail", starts="identify")
        selected = fake_arm.grab_selected()
        disp=None
        for module in solvers:
            acc, img = solvers[module].identify(fake_arm)
            if do_show:
                # todo: draw these more sensibly
                # lump them all together?
                if img is not None:
                    scaled = cv2.resize(img, (100,100))
                    disp = hstack_pad(disp, scaled)#, f"{module}", wait_forever=False)
            if acc:
                log.info(f"{module} identified - confidence: {acc}")
                dump = np.hstack((selected, img))
                #if not self.safe: dump_image(dump, dir="ident", starts=module)
                yield (acc, module)
        if do_show:
            display(disp)


    def watch(self):
        im = self.arm.grab_selected()
        ims = [im]
        while 1:
            im = self.arm.grab_selected()
            for base in ims:
                if np.sum(cv2.absdiff(im, base)) < 100 * 255:
                    break
            else:
                ims.append(im)
                dump_image(im,"watched")

def main():
    s = screen.Screen(2)
    r = Robot(s)
    #r.watch()
    #  r.defuse_bomb()
    try:
        # print(1/0)
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

def test():
    #s = screen.Screen(image_path="img2.bmp")
    #if 1:
    for im in images_in("dump/screens/", starts="unf"):
        #fake = screen.Screen(image_path="dump/watched/img49.bmp")
        #fake = screen.Screen(image_path="dump/test/img38.bmp")
        fake = screen.Screen(image=im)
        r = Robot(fake, safe=True)
        arm = r.arm
        list(arm.scan_modules())
        #img = arm.grab_module_unfocused(3)
        #import clock
        #clock.isClock(img)
        #cv2.waitKey(0)
        #return
        #im = r.arm.grab() #_selected()
        #r.arm.goto(1, after=0)
        #for i in range(6):
        #img = np.hstack(r.arm.grab_other_module(i) for i in range(6))
        #display(img, "module {}") #, wait_forever=False)
            #s1=screen.Screen(image=img)
            #display(img)
            #r1 = Robot(s1)
            #for mod in solvers:
            #    print(f"{mod} | ")
            #    if solvers[mod].identify(r.arm)[0]: print(f"{mod} | ")
            #    pass
      # display(im)
        # r = Robot(screen.Screen(2))
        #r.arm.wake_up()
        #time.sleep(0.2)
        #r.arm.goto(5)
     #  p=solvers["button"]#
     #  i, im = p.identify(r.arm)
     #  if i > 50:
     #      print(f"{i} | {p.new(r.arm)}")
     #      display(im)
     #  p.new(r.arm).solve()
        #p.test()
        #p.train(r.arm)
        #r.identify(True)
       # cv2.waitKey(0)
       # cv2.destroyAllWindows()
       # return
        #return
        #continue



if __name__ == "__main__":
    #test()
    main()

#TODO: draw clock
#TODO: draw on identify
#TODO: draw morse better
#todo: detect dark and redlight