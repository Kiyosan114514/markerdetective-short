import cv2
import numpy as np
import os

#local lib
import calculation
from FrameDivision import FrameDivision

__N = 32
__dt = 1 / 30
__win = np.hanning(32).astype(np.float32)
__freq = np.linspace(0, 1.0 / __dt, __N)

class diff():
    __VIDEO_DIR = 'video/'
    __VIDEO_EXT = '.h264'
    
    def different(self,):

