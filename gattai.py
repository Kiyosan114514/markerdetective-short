import cv2
import sys
import numpy as np

#動画と画像のパス作成
#video_path = "home/pi/Marker-Detection/video/power3w.h264"
#img_path = "home/pi/Marker-Detection/output/power3w.png"

#ファイル読み込み
cap1 = cv2.VideoCapture('/home/pi/marker/video/3m-0do.h264')
cap2 = cv2.imread('/home/pi/marker/output/3m-0do.png')

#新たに作成する動画ファイルの設定
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('/home/pi/marker/gattai/gattai3m-0do.mp4',fourcc, 20.0, (1024,768))

#ファイルが開いたか確認
if not cap1.isOpened():
    print("Video Error")
    sys.exit()

#if not cap2.isOpened():
    #print("Image Error")
    #sys.exit()

#出力動画のサイズ
width = 800
height = 600

cap2 = cv2.resize(cap2, (1024, 768))

cap2_2=cv2.cvtColor(cap2,cv2.COLOR_BGR2GRAY)

# 輪郭検出
contours, hierarchy = cv2.findContours(cap2_2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# リストの中の１つ目の長方形を取得
# リストには検出された長方形がcontours[i]に格納されているが１つしかないはずなのでその１つ目(インデックス番号的には0)を取得する

# 1個以上検出された場合
#if not np.array(contours).shape[0] == 1:
#    return 0, 0
# 検出されたオブジェクトが1つ飲みの場合，それを代入
a = contours[0]

#x座標とy座標をそれぞれ抽出する.ただし, aはndarray
x = a[:, :, 0]
y = a[:, :, 1]

# 検出された長方形の中心座標を計算する
X1 = (np.amax(x)).astype(np.int32)-30
X2 = (np.amin(x)).astype(np.int32)+30
Y1 = (np.amax(y)).astype(np.int32)+30
Y2 = (np.amin(y)).astype(np.int32)-30
Xo = ((X1 + X2) / np.int32(2)).astype(np.int32)
Yo = ((Y1 + Y2) / np.int32(2)).astype(np.int32)

cv2.line(cap2, (X1,Y1), (X2,Y1), (200,200,0),4)
cv2.line(cap2, (X1,Y1), (X1,Y2), (200,200,0),4)
cv2.line(cap2, (X1,Y2), (X2,Y2), (200,200,0),4)
cv2.line(cap2, (X2,Y1), (X2,Y2), (200,200,0),4)

while True:
    #1つ目の動画から1フレーム取得する
    ret1, frame1 =cap1.read()
    if not ret1:
        break
    
    frame2 = cap2

    #frame1 = cv2.resize(frame1, (600, 600))
    '''cv2.line(frame1, (X1,Y1), (X2,Y1), (200,200,0),4)
    cv2.line(frame1, (X1,Y1), (X1,Y2), (200,200,0),4)
    cv2.line(frame1, (X1,Y2), (X2,Y2), (200,200,0),4)
    cv2.line(frame1, (X2,Y1), (X2,Y2), (200,200,0),4)'''
    #frame2 = cv2.resize(frame2, (800, 600))

    #画像の重ね合わせ
    
    frame3 = cv2.addWeighted(
        src1=frame1, alpha=0.8,
        src2=frame2, beta=0.2,
        gamma=0
    )

    video.write(frame3)

video.release()
print('written')