# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import time

# --- Local ---
import calculation
from FrameDivision import FrameDivision

class CentralRecognition() :

    __OUTPUT_DIR = 'output/'
    __IMG_EXT = '.png'

    __N = 32
    __dt = 1 / 30

    __win = np.hanning(__N).astype(np.float32)
    __freq = np.linspace(0, 1.0 / __dt, __N)

    def __init__(self) :
        self.frame_divider = FrameDivision()
        if not os.path.exists(self.__OUTPUT_DIR) :
            os.makedirs(self.__OUTPUT_DIR)

    def getFrames(self) :
        self.frames = self.frame_divider.Frame_Division(self.VIDEO_NAME)
        self.frames2gray_img()

    def frames2gray_img(self) :
        self.img1 = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in self.frames])
        self.output = np.zeros((self.img1.shape[1], self.img1.shape[2]), dtype=np.uint8)
        self.different()
    
    def different(self):
        self.diff = np.zeros((self.img1.shape[1], self.img1.shape[2]), dtype=np.uint8)
        i=0
        for frame in self.frames:
            if i==31:
                break
            self.bef = self.img1[i, :, :]
            self.aft = self.img1[i+1, : ,:]
            self.diff_frame = cv2.absdiff(self.bef, self.aft)
            ret, diff_bin = cv2.threshold(self.diff_frame, 100, 255, 0)
            #cv2.imwrite('/home/pi/marker/dif/pic'+str(i)+'.png', diff_bin)
            self.diff=self.diff+diff_bin
            i=i+1

        a=np.where(self.diff>200)
        cv2.imwrite('/home/pi/marker/dif/diff-test3.png',self.diff)

        self.xfM = (np.amax(a[1])).astype(np.uint32)
        self.xfm = (np.amin(a[1])).astype(np.uint32)
        self.yfM = (np.amax(a[0])).astype(np.uint32)
        self.yfm = (np.amin(a[0])).astype(np.uint32)

        self.f_frames = np.array([frame[ self.yfm-30 : self.yfM+30 , self.xfm-30 : self.xfM+30 ]for frame in self.frames])
        self.img = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in self.f_frames])
        self.window_func()

    def window_func(self) :
        self.img = self.img.transpose(1, 2, 0)
        self.img = self.img * self.__win
        self.img = self.img.transpose(2, 0, 1)
        self.acf = 1/(self.__win.sum()/self.__N)
        self.fft()

    def fft(self) :
        self.fft_signal = np.fft.fft(self.img, axis=0)
        self.bpf()

    def bpf(self) :
        if self.MARKER_COLOR == 'Green' :
            self.filter_ = np.logical_and(self.__freq < 5.3, self.__freq > 4.8)
        elif self.MARKER_COLOR == 'Red' :
            self.filter_ = np.logical_and(self.__freq < 7.8, self.__freq > 6.8)
        self.fft_signal_filtered = self.fft_signal[self.filter_]
        self.normalize()

    def normalize(self) :
        self.fft_signal_filtered_amp = self.acf * np.abs(self.fft_signal_filtered)
        self.fft_signal_filtered_amp = self.fft_signal_filtered_amp / (self.__N / 2)
        self.fft_signal_filtered_amp[0, :, :] /= 2
        self.amp2pixel_value()

    def amp2pixel_value(self) :
        self.output_diff = np.max(self.fft_signal_filtered_amp, axis=0).astype(np.uint8)
        self.binarization()

    def binarization(self) :
        self.output_diff[self.output_diff.max() > self.output_diff] = 0
        self.output_diff[self.output_diff.max() <= self.output_diff] = 255
        self.output_result()

    def output_result(self) :
        y_offset = self.yfm-30
        x_offset = self.xfm-30
        self.output[y_offset:y_offset+self.output_diff.shape[0], x_offset:x_offset+self.output_diff.shape[1]] = self.output_diff
        self.output = cv2.resize(self.output, (3280, 2464))
        cv2.imwrite(os.path.join(self.__OUTPUT_DIR, self.VIDEO_NAME + self.__IMG_EXT), self.output)

    def central_recognition(self, VIDEO_NAME : str, MARKER_COLOR : str) :
        self.VIDEO_NAME = VIDEO_NAME
        self.MARKER_COLOR = MARKER_COLOR
        self.getFrames()
        return self.output

    def __del__(self):
        del self.frame_divider

if __name__=="__main__":
    start_time = time.time()
    centralRecognizer = CentralRecognition()
    VIDEO_NAME = input('Enter The Video file name (The Video Exists In The Video Folder.)')
    centralRecognizer.central_recognition(VIDEO_NAME, 'Green')
    print(time.time() - start_time)