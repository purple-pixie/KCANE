
from util import *
import logging
log = logging.getLogger(__name__)

def get_segments(image):
    for state in False, True:
        yield from get_segments_by_state(image, state)

def get_segments_by_state(image, lit = True):
    im = image.copy()
    if lit:
        hsv = to_hsv(image)
        h = hsv[..., 0]
        h[h < 5] = 178
        mask = inRangePairs(hsv, [(121, 179), (21, 255), (0, 255)]) > 0
        im[~mask] = 0
        gray = to_gray(im)
        gray = otsu(gray)
    else:
        gray = inRangePairs(im, [(22, 34), (23, 33), (4, 38)])
    gray = cv2.erode(gray, (5, 5), iterations=5)
    for cnt in contour(gray):
        cnt = Contour(cnt)
        #small edge ~5
        #long edge ~10 or ~17
        if (3 < cnt.w < 7 and 14 < cnt.h < 30) or (2 < cnt.h < 6 and 6 < cnt.w < 14):
            yield lit, cnt
        #else: print(cnt)

#def a*()
def sort_by_y(flag_contour):
    flag, contour = flag_contour
    return contour.centre[1]

def partial_digit(segment, start, stop):
    return [flag for flag, cnt in sorted(segment[start:stop], key=sort_by_y)]

def get_digit(segment):
    """get digit from 7 lit flag, contour pairs"""
    #TODO: do this
    lit_flags = partial_digit(segment, 0, 2) + partial_digit(segment, 2, 5) + partial_digit(segment, 5,7)
    if lit_flags in digit_masks:
        return digit_masks.index(lit_flags)
    return 8

digit_masks = [
    [1,1,1,0,1,1,1],
    [0,0,0,0,0,1,1],
    [0,1,1,1,1,1,0],
    [0,0,1,1,1,1,1],
    [1,0,0,1,0,1,1],
    [1,0,1,1,1,0,1],
    [1,1,1,1,1,0,1],
    [0,0,1,0,0,1,1],
    [1,1,1,1,1,1,1],
    [1,0,1,1,1,1,1]
]

def isClock(image):
    # for i, cnt in enumerate(contour(gray)):
    #     x1, y1 = cnt[0][0] * 4
    #     draw_label(canvas, (x1, y1 - 40), f"{i}", color=(0, 0, 0))
    # actual isClock:
    #display(image, "clock_", wait_forever=False)
    #time.sleep(0.5)
    #dump_image(image, dir="clock", starts=f"length {len(list(get_segments(image)))}_")
    segs = list(get_segments(image))
    log.debug(f"{len(segs)} lcd-segments found")
    return len(segs) == 28

def read_clock(image):
    #get all 28 segments (7 per digit) and sort them by x-coord
    segments = sorted(get_segments(image), key=lambda x: x[1].centre)
    segments = np.array(segments, dtype=object)
    try:
        #split them up into 4 discrete digits
        segments = segments.reshape((4,7,2))
    except ValueError:
        #can't reshape to 4,7,2 means it wasn't 28,2 so we haven't found a good LCD
        return "", None

    #drawing fluff
    canvas = np.full(image.shape, 120, "uint8")
    for digit in segments:
        for state, cnt in digit:
            if state:
                cnt.draw(canvas, detail_color=(0,0,255), detail_width=-1)
            else:
                cnt.draw(canvas, detail_color=(50,50,50))
        #dump = np.hstack((canvas, image))
        #print(f"digit is: {get_digit(digit)}")
        #display(dump)
    output =f"{get_digit(segments[0])}{get_digit(segments[1])}:{get_digit(segments[2])}{get_digit(segments[3])}"
    #print(output)
    #display(image)
    return output, canvas