from glob import glob
import cv2
import numpy as np

import ctypes as C
libbmog = C.cdll.LoadLibrary('./libs/bmog/libbmog.so')


class BMOG(object):
    def __init__(self, nmixtures=5, threshold_l=35, threshold_a=12, threshold_b=12, background_ratio=1.0, postprocessing_size=9):
        #print("init bmog..")

        libbmog.init_bgs.argtypes = [
            C.c_int, C.c_int, C.c_int, C.c_int, C.c_double, C.c_int]
        libbmog.init_bgs(nmixtures, threshold_l, threshold_a,
                          threshold_b, background_ratio, postprocessing_size)

    def apply(self, img):
        # lib.apply(self.obj, img)
        (rows, cols) = (img.shape[0], img.shape[1])
        res = np.zeros(dtype=np.uint8, shape=(rows, cols))
        libbmog.getfg(img.shape[0], img.shape[1],
                      img.ctypes.data_as(C.POINTER(C.c_ubyte)),
                      res.ctypes.data_as(C.POINTER(C.c_ubyte)))
        return res

    
    def get_bg(self, img_shape):
        print("getting bg!")
        (rows, cols) = (img.shape[0], img.shape[1])
        res = np.zeros(dtype=np.uint8, shape=(rows, cols))
        libbmog.getbg(img.shape[0], img.shape[1],
                      res.ctypes.data_as(C.POINTER(C.c_ubyte)))
        return res



# example usage
if __name__ == "__main__":
    bgs = BMOG()
    image_dir = "/home/moritz/Downloads/test_sequences/01_day-fog/*.jpg"
    for img_path in sorted(glob(image_dir)):
        img = cv2.imread(img_path)
        fg = bgs.apply(img)
        cv2.imshow("img", img)
        cv2.imshow("fg", fg)
        cv2.waitKey(1)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

        #bg = bgs.get_bbg(fg)
        #cv2.imshow("bg", bg)
        #cv2.waitKey(1)

        #if cv2.waitKey(1) and 0xFF == ord('q'):
        #    break
