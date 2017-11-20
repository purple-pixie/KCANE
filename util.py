import cv2

def read_key():
    key = 256
    while key > 127:
        key = cv2.waitKey(0)
    return key&0xFF


def display(c, text="image"):
    cv2.imshow(text, c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_and_display(c, text="input a key"):
    cv2.imshow(text, c)
    return read_key()