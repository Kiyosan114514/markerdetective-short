import cv2
import numpy as np

def find_rect_of_target_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[((h < 20) | (h > 200)) & (s > 128)] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        approx = cv2.convexHull(contour)
        rect = cv2.boundingRect(approx)
        rects.append(np.array(rect))
    return rects

if __name__ == "__main__":
    #動画の取得
    capture = cv2.VideoCapture('/home/pi/marker/video/power-red-3m.h264')

    #作成する動画の設定
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter('/home/pi/marker/power-red-3m-kensyutu.mp4',fourcc, 20.0, (1024,768))

    while cv2.waitKey(30) < 0:
        _, frame = capture.read()
        rects = find_rect_of_target_color(frame)
        if len(rects) > 0:
            rect = max(rects, key=(lambda x: x[2] * x[3]))
            cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (0, 0, 255), thickness=2)
        #cv2.imshow('red', frame)
        video.write(frame)
    capture.release()
    cv2.destroyAllWindows()