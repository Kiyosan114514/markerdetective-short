import cv2
import numpy as np
import os
import time

class Calc():

    # Principal Point
    __Cx = 1633.39414024524
    __Cy = 1218.19031757266

    # Height from floor to marker[cm]
    __H1 = 220

    # Height from floor to camera[cm]
    __H2 = 9.5

    # Height from camera to marker[cm]
    __H = __H1 - __H2

    # Radial strain coefficient
    __K1 = -3.252646756218603437e-15
    __K2 = 1.312609922083992325e-11
    __K3 = -2.146160690813455963e-08
    __K4 = 1.817365729797309660e-05
    __K5 = -8.396836335535312612e-03
    __K6 = 2.080660359742307897e+00
    __K7 = -1.899798432998080386e+02

    __MARK_DIR = 'mark/'
    __MARK_EXT = '.png'

    def __init__(self) :
        if not os.path.exists(self.__MARK_DIR) :
            os.makedirs(self.__MARK_DIR)

    def __get_center_point(self, result) :
        # Extraction of contours
        contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If only one contour is detected, Extraction of x-point and y-point
        contour = contours[0]
        marker_x = np.array(contour)[:, :, 0]
        marker_y = np.array(contour)[:, :, 1]

        # Calclate the center point of the detected contour
        marker_x_1 = (np.amax(marker_x)).astype(np.float32)
        marker_x_2 = (np.amin(marker_x)).astype(np.float32)
        marker_y_1 = (np.amax(marker_y)).astype(np.float32)
        marker_y_2 = (np.amin(marker_y)).astype(np.float32)
        self.marker_x_center = ((marker_x_1 + marker_x_2) / np.float32(2)).astype(np.int32)
        self.marker_y_center = ((marker_y_1 + marker_y_2) / np.float32(2)).astype(np.int32)

    def __get_difference_from_the_center_point(self) :
        diff_x = abs(self.marker_x_center - self.__Cx)
        diff_y = abs(self.marker_y_center - self.__Cy)
        self.r = np.power(np.power(diff_x, 2) + np.power(diff_y, 2), 0.5)

    def __angle(self, rad) :
        if rad < 0 :
            rad = 2 * np.pi + rad
        return rad

    def __get_zenith_angle(self) :
        self.zenith_angle \
        = np.deg2rad( \
            self.__K1 * self.r ** 6 + \
            self.__K2 * self.r ** 5 + \
            self.__K3 * self.r ** 4 + \
            self.__K4 * self.r ** 3 + \
            self.__K5 * self.r ** 2 + \
            self.__K6 * self.r ** 1 + \
            self.__K7 \
            )

    def __get_azimuth_angle(self) :
        x = self.marker_x_center - self.__Cx
        y = self.marker_y_center - self.__Cy
        X = x * np.cos(np.pi / 2) + y * np.sin(np.pi / 2)
        Y = y * np.cos(np.pi / 2) - x * np.sin(np.pi / 2)
        self.azimuth_angle = np.arctan2(Y, X)
        self.phi = self.__angle(self.azimuth_angle)

    def __calc_distance(self) :
        marker_x = self.__H * np.tan(self.zenith_angle) * np.cos(self.azimuth_angle)
        marker_y = self.__H * np.tan(self.zenith_angle) * np.sin(self.azimuth_angle)
        self.distance = np.power(np.power(marker_x, 2) + np.power(marker_y, 2), 0.5)

    def draw_marker(self, result) :
        self.marker_detection_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        cv2.circle(self.marker_detection_img, center=(self.marker_x_center, self.marker_y_center), radius=50, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_4, shift=0)
        cv2.imwrite(os.path.join(self.__MARK_DIR, self.VIDEO_NAME + self.__MARK_EXT), self.marker_detection_img)

    def get_distance_and_phi(self, result, VIDEO_NAME) :
        self.error = 0
        self.t3 = time.time()
        self.VIDEO_NAME = VIDEO_NAME
        self.__get_center_point(result)
        self.__get_difference_from_the_center_point()
        self.__get_zenith_angle()
        self.__get_azimuth_angle()
        self.__calc_distance()
        self.draw_marker(result)
        self.t4 = time.time()
        self.calcT = self.t4-self.t3
        return self.distance, self.phi