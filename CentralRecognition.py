 #-*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import time

# --- Local ---
import calculation
from FrameDivision import FrameDivision

class CentralRecognition() :

    __OUTPUT_DIR = 'output/'
    __FFT_DIR = 'fft/'
    __DIF_DIR = 'dif/'
    __DIFF_DIR = 'diff/'
    __IMG_EXT = '.png'

    __N = 32
    __dt = 1 / 30

    __win = np.hanning(__N).astype(np.float32)
    __freq = np.linspace(0, 1.0 / __dt, __N)

    def __init__(self) :
        self.frame_divider = FrameDivision()
        if not os.path.exists(self.__OUTPUT_DIR) :
            os.makedirs(self.__OUTPUT_DIR)

    #映像をフレーム分割
    def __getFrames(self) :
        self.frames = self.frame_divider.Frame_Division(self.VIDEO_NAME)

    #グレースケール化
    def __frames2gray_img(self) :
        self.img = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in self.frames])

    #差分の抽出
    def different(self):
        self.diff = np.zeros((self.img.shape[1], self.img.shape[2]), dtype=np.uint32)
        i=0
        for frame in self.frames:
            if i==31:
                break
            self.bef = self.img[i, :, :]
            self.aft = self.img[i+1, : ,:]
            self.bef = cv2.equalizeHist(self.bef)
            self.aft = cv2.equalizeHist(self.aft)
            self.diff_frame = cv2.absdiff(self.bef, self.aft)
            self.diff=self.diff+self.diff_frame
            i=i+1
        
        #差分を正規化
        self.diff = (self.diff/self.diff.max())*255
        self.diff = self.diff.astype(np.uint8)
        cv2.imwrite(os.path.join(self.__DIF_DIR, self.VIDEO_NAME + self.__IMG_EXT), self.diff) 

        #差分を２値化
        ret, self.diff = cv2.threshold(self.diff, 0, 255, cv2.THRESH_OTSU)

        #カメラの枠外を除去
        self.mask = np.zeros((self.img.shape[1], self.img.shape[2]), dtype=np.uint8)
        cv2.circle(self.mask, center=(self.img.shape[2] // 2, self.img.shape[1] // 2), radius=self.img.shape[1]//2, color=255, thickness=-1)
        self.diff[self.mask == 0] = 0

        cv2.imwrite(os.path.join(self.__DIFF_DIR, self.VIDEO_NAME + self.__IMG_EXT), self.diff)

    #窓関数に入れる
    def __window_func(self) :
        self.img = self.img.transpose(1, 2, 0)
        #差分部分を切り取り
        self.masked_img = self.img[self.diff == 255]
        print(self.masked_img.shape)
        self.img = self.img.transpose(2, 0, 1)
        self.masked_img = self.masked_img * self.__win     
        self.acf = 1/(self.__win.sum()/ self.__N)
        self.masked_img = self.masked_img.transpose(1,0)

    #FFT
    def __fft(self) :
        self.fft_signal = np.fft.fft(self.masked_img, axis=0)

    #BPF
    def __bpf(self) :
        if self.MARKER_COLOR == 'Green' :
            self.filter_ = np.logical_and(self.__freq < 5.3, self.__freq > 4.8)
        elif self.MARKER_COLOR == 'Red' :
            self.filter_ = np.logical_and(self.__freq < 7.8, self.__freq > 6.8)
        elif self.MARKER_COLOR == 'Blue' :
            self.filter_ = np.logical_and(self.__freq < 12.5, self.__freq > 11.6)
        self.fft_signal_filtered = self.fft_signal[self.filter_]

    #正規化
    def __normalize(self) :
        self.fft_signal_filtered_amp = self.acf * np.abs(self.fft_signal_filtered)
        self.fft_signal_filtered_amp = self.fft_signal_filtered_amp / (self.__N / 2)
        self.fft_signal_filtered_amp[0, :] /= 2

    #FFTした結果を画像に変換
    def __amp2pixel_value(self) :
        self.output = np.zeros_like(self.img[:1])
        self.output = self.output.transpose(1,2,0)
        self.output[self.diff == 255] = self.fft_signal_filtered_amp.transpose(1,0)
        self.output = self.output.transpose(2,0,1)
        self.output = np.max(self.output, axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(self.__FFT_DIR, self.VIDEO_NAME + self.__IMG_EXT), self.output)

    #二値化
    def __binarization(self) :
        # 最大値を取得(探査用)
        self.output[self.output.max() > self.output] = 0
        self.output[self.output.max() <= self.output] = 255

        # 判別分析法(Debug用)
        #ret, self.output = cv2.threshold(self.output, 0, 255, cv2.THRESH_OTSU)

    def __output_result(self) :
        self.output = cv2.resize(self.output, (3280, 2464))
        cv2.imwrite(os.path.join(self.__OUTPUT_DIR, self.VIDEO_NAME + self.__IMG_EXT), self.output)

    def central_recognition(self, VIDEO_NAME : str, MARKER_COLOR : str) :
        self.VIDEO_NAME = VIDEO_NAME
        self.MARKER_COLOR = MARKER_COLOR
        self.__getFrames()
        self.__frames2gray_img()
        self.different()
        self.__window_func()
        self.__fft()
        self.__bpf()
        self.__normalize()
        self.__amp2pixel_value()
        self.__binarization()
        self.__output_result()
        return self.output

    def __del__(self):
        del self.frame_divider

if __name__=="__main__":
    start_time = time.time()
    centralRecognizer = CentralRecognition()
    VIDEO_NAME = input('Enter The Video file name (The Video Exists In The Video Folder.)')
    centralRecognizer.central_recognition(VIDEO_NAME, 'Green')
    print(time.time() - start_time)