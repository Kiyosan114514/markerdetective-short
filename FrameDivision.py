import cv2
import os

class FrameDivision :
    __VIDEO_DIR = 'video/'
    __VIDEO_EXT = '.h264'

    def Frame_Division(self, STEP : str) :

        # Read Video
        cap = cv2.VideoCapture(os.path.join(self.__VIDEO_DIR, STEP + self.__VIDEO_EXT))
        #cap = cv2.VideoCapture('/home/pi/marker/video/new-test14-max.h264')
        #aaaaaaaaaa

        # Counter
        counter = 0

        # Frame Array
        frames = []

        while(cap.isOpened()) :

            # Frame Division
            result, frame = cap.read()

            if result == True :
                frames.append(frame)
                counter += 1

            else :
                break

        frames = frames[32:64]
        cap.release()
        return frames

if __name__ == '__main__' :
    frame_divider = FrameDivision()
    frame_divider.Frame_Division(input('Input Video Name : '))