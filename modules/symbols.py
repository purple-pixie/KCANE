from util import *
import itertools
import logging
from operator import itemgetter
log = logging.getLogger(__name__)
import robot_arm
from robot_arm import  sleep

class Symbols():
    def __init__(self, robot:robot_arm.RobotArm):
        self.robot = robot

    def solve(self):
        #TODO: prevent this from trying to iterate None when it misreads the symbols. Or perfect symbol identifying
        im = self.robot.grab_selected()
        canvas = np.full(im.shape,120,"uint8")
        symbol_positions = {match(cnt)[0]: cnt.centre for cnt, indicator in symbol_indicators(im, canvas)}
        self.robot.robot.draw_module(canvas)
        print(symbol_positions)
        ordered = self.get_order(symbol_positions)
        print(ordered)
        for symbol in ordered:
            log.info(f"clicking on {symbol.encode()} {symbol_positions[symbol]} ")
            self.robot.moduleto(*symbol_positions[symbol])
            self.robot.click(before=0.1, after=0.1)
           # print(self.robot.mouse_position())
            #time.sleep(1)


    def get_order(self, found):
        for column in keys:
            try:
                return sorted(found, key=lambda x: column.index(x[0]))
            except ValueError:
                continue

def match(cnt, mode=cv2.CONTOURS_MATCH_I1):
    if (cnt.w*cnt.h) < 100:
        return 'Ͽ', 0.
    symbol_dists = ((label, cv2.matchShapes(cnt.points, symbol,mode,0.)) for label, symbol in symbols.items())
    return sorted(symbol_dists, key=itemgetter(1))[0]
shapes = {}
symbols = {}
#keys=["ϘѦƛϞѬϗϿ","ӬϘϿҨ☆★ϗ¿","©ѼҨҖԆƛ☆★","б¶ѢѬҖ¿ټ","ΨټѢϾ¶Ѯ☆★","бӬ҂æΨҊΩ"]
keys=["ϘѦƛϞѬϗϿϾ","ӬϘϿϾҨ☆★ϗ¿","©ѼҨҖԆƛ☆★","б¶ѢѬҖ¿ټ","ΨټѢϿϾ¶Ѯ☆★","бӬ҂æΨҊΩ"]

class Solver():
    def __init__(self):
        self.read_saved()
        print(f"Recalled: {symbols.keys()}")
        pass

    def learn_from_manual(self):
        im = cv2.imread("data/modules/symbols/template.bmp")
        gray = 255 - to_gray(im)
        cnts = contour(gray, mode=cv2.RETR_EXTERNAL)
        c = [cnt for cnt in cnts if cv2.arcLength(cnt,False) > 100]
        for cnt in c:
            cv2.drawContours(im, [cnt], -1, (0,0,255), 1)
            x,y,*rest = cv2.boundingRect(cnt)
            x, y = x//130, y//91
            shapes[keys[x][y]] = cnt
        log.debug(f"{len(shapes)} symbols learned")
    def new(self, robot:robot_arm.RobotArm):
        return Symbols(robot)
    def identify(self, robot:robot_arm.RobotArm):
        img = robot.grab_selected()
        self.canvas = np.full(img.shape,120,"uint8")
        si = list(symbol_indicators(img, self.canvas))
        #print(si)
        #display(self.canvas, scale=5)
        return (len(si) == 4) * 200, self.canvas

    def read_saved(self):
        for filename in glob.glob("data/modules/symbols/*.npy"):
            label = naked_filename(filename)
            symbols[label] = np.load(filename)
    def save_symbols(self):
        for name, points in symbols.items():
            np.save(f"data/modules/symbols/{name}", points, allow_pickle=False)

    def train(self, robot:robot_arm.RobotArm):
        print(set("".join(keys))-set(symbols.keys()))
        canvas = np.full((140,140,3),120,"uint8")
        img = robot.grab_selected()
        si = list(symbol_indicators(img, None))
        for i, (cnt, indicator) in enumerate(si):
            cnt.draw(canvas, rect_color=(0,255,0))
            draw_label(canvas, indicator.centre, f"{i}")
            display(canvas, "symbol training", scale=5, wait_forever=False)
            cv2.waitKey(20)
            label = input(f"symbol {i} ({match(cnt)}): ")
            if label == "":
                continue
            if label not in symbols:
                #symbols.
                #learn symbol
                symbols[label] = cnt.points
            else:
                #it is already there, show diff?
                print(f"i already have {label} | match is {cv2.matchShapes(cnt.points, symbols[label], 1, 0.)}")
        self.save_symbols()

    def test(self, robot:robot_arm.RobotArm):
        img = robot.grab_selected()
        canvas = np.full((140,140,3),120,"uint8")
        can = canvas.copy()
        si = list(symbol_indicators(img, can))
       # display(can)
        for i, (cnt, indicator) in enumerate(si):
            cnt.draw(canvas, rect_color=(0,255,0))
            draw_label(canvas, indicator.centre, f"{i}")
        for i, (cnt, indicator) in enumerate(si):
            print(f"{i}: {match(cnt, 1)}")
        display(canvas, "symbol training", scale=5)


       #for i, pts in symbols.items():
       #    canvas = np.full((200,200,3),120,"uint8")
       #    c=Contour(pts)
       #    c.draw(canvas)
       #    print(i)
       #    display(canvas, scale=3)
       #pass


def symbol_indicators(img, can=None):
    mask = inRangePairs(to_hsv(img), [(16, 42), (16, 122), (189, 255)])
    display(mask,"mask",wait_forever=False)
    cnts, h = contour(mask, mode=cv2.RETR_TREE, return_hierarchy=True)
    nested_cnts = {}
    symbol="something went wrong"
    for idx, (cnt, data) in enumerate(zip(cnts, h)):
        parent = data[3]
        if parent <0:
            if can is not None: cv2.drawContours(can, [cnt], -1, (0,0,255),1)
            continue
        #ignore any grand-children
        if h[parent][3] >= 0:
            if can is not None: cv2.drawContours(can, [cnt], -1, (255,255,0),1)
            continue
        #symbol is the first child (i.e no previous)
        #in the case of Ͽ and Ͼ it will report the central dot first but that's fine
        #we don't actually need to distinguish between them (or between the two stars)
        if data[1] < 0:
            symbol = cnt
            if can is not None: cv2.drawContours(can, [cnt], -1, (0,255,0),1)
            continue
        #indicators are the last child
        if data[0] < 0:
            if can is not None: cv2.drawContours(can, [cnt], -1, (0,255,255),1)
            yield (Contour(symbol), Contour(cnt))
        if can is not None: cv2.drawContours(can, [cnt], -1, (255,255,255),1)
