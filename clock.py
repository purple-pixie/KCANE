
from util import *
from matplotlib import pyplot as plt

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
        if (3 < cnt.w < 7 and 14 < cnt.h < 30) or (2 < cnt.h < 6 and 8 < cnt.w < 14):
            yield lit, cnt
        #else: print(cnt)

#def a*()
def get_digit(segment):
    """get digit from 7 contours"""

def isClock(image):
    # for i, cnt in enumerate(contour(gray)):
    #     x1, y1 = cnt[0][0] * 4
    #     draw_label(canvas, (x1, y1 - 40), f"{i}", color=(0, 0, 0))
    # actual isClock:
    # segments = list(get_segments(image)
 #   return len(segments) == 28

#def read_clock(image):
    segments = sorted(get_segments(image), key=lambda x: x[1].centre)
    segments = np.array(segments, dtype=object)
    try:
        segments = segments.reshape((4,7,2))
    except ValueError:
        #can't reshape to 4,7,2 means it wasn't 28,2 so we haven't found a good LCD
        return ""
    canvas = np.full(image.shape, 120, "uint8")

    for state, cnt in segments[1]:
        if state:
            cnt.draw(canvas, detail_color=(0,0,255), detail_width=-1)
        else:
            cnt.draw(canvas, detail_color=(0,0,0))
    dump = np.hstack((canvas, image))
    display(dump)
    for d in range(4):
        pass
    #centres = np.array([c.centre for d, c in cnts])

    #dump_image(dump, dir="clock/fail")

if __name__ == "__main__":
    ims = images_in("", starts="", return_names=True)
    for name, im in ims:
        print(name)
        #im = im[400:,400:]
        hsv = to_hsv(im)
        h = hsv[..., 0]
        h[h < 5] = 178
        mask = inRangePairs(hsv, [(121, 179), (21, 255), (0, 255)]) > 0
        im[~mask] = 0
        gray = to_gray(im)
       # display(gray, "gray", wait_forever=False)
      #  gray = blur(gray, (5, 5))
      #  display(gray, "blur", wait_forever=False)
        gray = otsu(gray)
      #  display(gray, "otsu", wait_forever=False)
        gray = cv2.erode(gray, (5, 5), iterations=5)
     #   display(gray, "erode") #, wait_forever=False)
        canvas = np.full(im.shape, 120, "uint8")
        for i, cnt in enumerate(contour(gray)):
            op = cv2.arcLength(cnt, False)
            op = cv2.contourArea(cnt)
            col = (0,255,0)#random_color()
            if op < 40:
                col = (255,255,0)
            if op > 120:
                col = (0,0,255)
            cv2.drawContours(canvas, [cnt], -1, col, 1)
            x1, y1 = cnt[0][0]
            draw_label(canvas, (x1, y1 - 10), f"{i}", color=(0,0,0), font_scale=0.3)
            op = cv2.arcLength(cnt, False)
            cl = cv2.arcLength(cnt, True)
            print(f"contour {i:2} - len {op} | diff {op-cl} | area {cv2.contourArea(cnt)}")
        canvas = cv2.resize(canvas, None, fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
       # for i, cnt in enumerate(contour(gray)):
       #     x1, y1 = cnt[0][0] * 4
       #     draw_label(canvas, (x1, y1 - 40), f"{i}", color=(0, 0, 0))
#
        display(canvas)

        #contour  4 - len 27.899494767189026 | diff -8.0 | area 58.5
      # '' contour  5 - len 28.313708186149597 | diff -7.0 | area 56.0
      # '' contour  6 - len 32.3137081861496 | diff -3.0 | area 52.0
        #contour 15 - len 19.727921843528748 | diff -1.0 | area 24.5 - colon
        #contour 39 - len 18.899494767189026 | diff -1.0 | area 22.5 - colon


#[(19, 33), (192, 255), (149, 255)] - yellow button
#[(0, 33), (0, 255), (137, 255)] - white button !!also finds yellow, do after
#[(110, 120), (160, 255), (97, 255)] - blue button

#[(174, 179), (73, 255), (163, 255)] - red button

    