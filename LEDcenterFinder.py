# -*- coding: utf-8 -*-
import math
import cv2
import numpy as np
import threading
import time
import os
import sys

# --- Local ---
from CentralRecognition import CentralRecognition
from calculation import Calc
from video import Video

class LEDCenterFinder() :

    def __init__(self) :
        self.video_cap = Video()
        self.central_recognizer = CentralRecognition()
        self.calculator = Calc()

    def __video_capture(self) :
        self.video_cap.capture(self.VIDEO_NAME)

    # Detects blinking marker
    def __marker_detection(self) :
        self.__marker_detection_img = self.central_recognizer.central_recognition(self.VIDEO_NAME, self.MARKER_COLOR)

    # Calculate the distance and angle to the blinking marker
    def __calc_distance_and_phi(self) :
        self.__distance, self.__phi = self.calculator.get_distance_and_phi(self.__marker_detection_img, self.VIDEO_NAME)

    # Return distance[m] and angle(phi)[deg]
    def getRTheta(self, VIDEO_NAME : int, MARKER_COLOR : str) :
        self.VIDEO_NAME = str(VIDEO_NAME)
        self.MARKER_COLOR = MARKER_COLOR
        self.__video_capture()
        self.__marker_detection()
        self.__calc_distance_and_phi()
        return self.__distance * 0.01, np.rad2deg(self.__phi)

if __name__ == '__main__' :
    x = LEDCenterFinder()
    VIDEO_NAME = input('VIDEO NAME : ')
    MARKER_COLOR = input('MARKER COLOR : ')
    t1=time.time()
    print(x.getRTheta(VIDEO_NAME, MARKER_COLOR))
    t2=time.time()
    T=t2-t1
    print(f"経過時間：{T}")