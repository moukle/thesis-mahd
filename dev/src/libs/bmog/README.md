# BMOG
Code is from `Martins, Isabel et al` for their paper **BMOG: Boosted Gaussian Mixture Model with Controlled Complexity"** (`ISBN: 978-3-319-58838-4`)

## Steps to include BMOG in Python
### Extend `bmog.hpp`
``` cpp
Ptr<BackgroundSubtractorBMOG> bgBMOG;

extern "C" void getfg(int rows, int cols, unsigned char* imgData,
    unsigned char *fgD) {
    cv::Mat img(rows, cols, CV_8UC3, (void *) imgData);
    cv::Mat fg(rows, cols, CV_8UC1, fgD);
    bgBMOG->apply(img, fg);
}

extern "C" void init_bgs(int nmixtures, int threshold_L, int threshold_a, int threshold_b, double backgroundRatio, int postProcessingSize) {
    bgBMOG = createBackgroundSubtractorBMOG();
    
    // defaults
    // bgBMOG->setNMixtures( 5 );
    // bgBMOG->setVarThreshold_L( 35 );
    // bgBMOG->setVarThreshold_a( 12 );
    // bgBMOG->setVarThreshold_b( 12 );
    // bgBMOG->setPostProcessingSize( 9 );
    // bgBMOG->setBackgroundRatio( 1.0 );
    // ----
    //bgBMOG->setTransientFrames( 50 );
    //bgBMOG->setVarThresholdGen( 8.0 );
    //bgBMOG->setVarInit( 11.0 );
    //bgBMOG->setHistory( 100 );
    //bgBMOG->setComplexityReductionThreshold( 0.05 );
    
    // set new BMOG parameters
    bgBMOG->setNMixtures(nmixtures);
    bgBMOG->setVarThreshold_L(threshold_L);
    bgBMOG->setVarThreshold_a(threshold_a);
    bgBMOG->setVarThreshold_b(threshold_b);
    bgBMOG->setBackgroundRatio(backgroundRatio);
    bgBMOG->setPostProcessingSize(postProcessingSize);
}
```

### Create shared Lib
``` sh
g++ BMOG.cpp `pkg-config opencv --libs --cflags` \
        -c -fPIC -o BMOG.o

g++ -shared -Wl,-soname,libbmog.so \
	-o libbmog.so BMOG.o \
	`pkg-config opencv --libs --cflags`
```

### Create Python Wrapper
``` py
from glob import glob
import cv2
import numpy as np

import ctypes as C
libbmog = C.cdll.LoadLibrary('./libs/bmog/libbmog.so')


class BMOG(object):
    def __init__(self, nmixtures=5, threshold_l=35, threshold_a=12, threshold_b=12, background_ratio=1.0, postprocessing_size=9):
        print("init bmog..")

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

```

``` py
# example usage
if __name__ == "__main__":
    bgs = BMOG()
    for img_path in sorted(glob("/home/ritzo/Documents/Bachelor/data/matchbox/00_sequence-day-multi/images_rain/*.jpg")):
        img = cv2.imread(img_path)
        fg = bgs.apply(img)
        cv2.imshow("img", img)
        cv2.imshow("fg", fg)
        cv2.waitKey(1)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
```
