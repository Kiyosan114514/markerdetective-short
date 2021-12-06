import cv2
import sys
import numpy as np
import time

__N = 32
__dt = 1 / 30
__win = np.hanning(32).astype(np.float32)
__freq = np.linspace(0, 1.0 / __dt, __N)
t1=time.time()

#ファイル読み込み
cap1 = cv2.VideoCapture('/home/pi/marker/video/3m-0do.h264')

#ファイルが開いたか確認
if not cap1.isOpened():
    print("Video Error")
    sys.exit()

#フレーム分割
frames = []
while(cap1.isOpened()):
    ret,frame =cap1.read()
    if ret == True :
        frames.append(frame)
    
    else:
        break
    
frames= frames[32:64]

#差分出力設定
diff =np.zeros((768,1024),dtype=np.uint8)

#差分抽出
gray_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
i=0
n=0
for frame in frames:
    if i==31:
        break
    bef = gray_frames[i, :, :]
    aft = gray_frames[i+1, : ,:]
    diff_frame = cv2.absdiff(bef, aft)
    """bef=bef.astype(np.float32)
    aft = aft.astype(np.float32)
    diff_frame = bef-aft
    diff_frame = np.where(diff_frame<0, 0 , diff_frame)
    diff_frame = diff_frame.astype(np.uint8)"""
    ret, diff_bin = cv2.threshold(diff_frame, 100, 255, 0)
    """check = np.any(diff_bin == 255)
    if check == True:
        diff=diff+diff_bin
        n=n+1"""
    diff=diff+diff_bin
    i=i+1

cv2.imwrite('/home/pi/marker/dif/3m-0dodiff.png',diff)
print(aft.dtype)
#print(f"使った枚数：{n}")

#差分輪郭抽出
#contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i=0
"""
#差分中心座標獲得
x_centers=[]
y_centers=[]
while i in range(len(contours)):
    contour = contours[i]
    x = np.array(contour)[:, :, 0]
    y = np.array(contour)[:, :, 1]

    # Calclate the center point of the detected contour
    x_1 = (np.amax(x)).astype(np.float32)
    x_2 = (np.amin(x)).astype(np.float32)
    y_1 = (np.amax(y)).astype(np.float32)
    y_2 = (np.amin(y)).astype(np.float32)
    x_center = ((x_1 + x_2) / np.float32(2)).astype(np.uint32)
    y_center = ((y_1 + y_2) / np.float32(2)).astype(np.uint32)
    x_centers.append(x_center)
    y_centers.append(y_center)
    i=i+1"""

#print(np.where(diff>200))
a=np.where(diff>200)
#print(a[0])

print('written')
print (gray_frames.shape)
#print(hierarchy.shape)

#差分が生じている部分を切り取り
xfM = (np.amax(a[1])).astype(np.uint32)
xfm = (np.amin(a[1])).astype(np.uint32)
yfM = (np.amax(a[0])).astype(np.uint32)
yfm = (np.amin(a[0])).astype(np.uint32)
print(xfM)
print(xfm)
print(yfM)
print(yfm)

#print(f_frames)
f_frames = np.array([frame[ yfm : yfM , xfm : xfM ]for frame in frames])
gray_f_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in f_frames])
print(gray_f_frames.shape)

output = np.zeros((768,1024),dtype=np.uint8)

#窓関数をかける
img = gray_f_frames.transpose(1, 2, 0)
img = img * __win            
img = img.transpose(2, 0, 1)
acf = 1/(__win.sum()/ __N)

#fft処理
fft_signal = np.fft.fft(img, axis=0)

#bpfに通す
filter_ = np.logical_and(__freq < 5.3, __freq > 4.8)
fft_signal_filtered = fft_signal[filter_]

#正規化
fft_signal_filtered_amp = acf * np.abs(fft_signal_filtered)
fft_signal_filtered_amp = fft_signal_filtered_amp / (__N / 2)
fft_signal_filtered_amp[0, :, :] /= 2

#
output_frame = np.max(fft_signal_filtered_amp, axis=0).astype(np.uint8)

#2値化
output_frame[output_frame.max() > output_frame] = 0
output_frame[output_frame.max() <= output_frame] = 255

#出力画像と切り抜き画像の重ね合わせ
y_offset = yfm
x_offset = xfm
output[y_offset:y_offset+output_frame.shape[0], x_offset:x_offset+output_frame.shape[1]] = output_frame
output = cv2.resize(output, (3280, 2464))
cv2.imwrite('/home/pi/marker/output/dif.png',output)
t2=time.time()
T=t2-t1
print(f"経過時間：{T}")