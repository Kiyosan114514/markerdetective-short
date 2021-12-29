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
cap1 = cv2.VideoCapture('/home/pi/marker/video/asdfg.h264')

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
t2=time.time()

#差分出力設定
diff =np.zeros((768,1024),dtype=np.uint32)

#差分抽出
img = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
i=0
n=0
for frame in frames:
    if i==31:
        break
    bef = img[i, :, :]
    aft = img[i+1, : ,:]
    bef = cv2.equalizeHist(bef)
    aft = cv2.equalizeHist(aft)
    diff_frame = cv2.absdiff(bef, aft)
    diff=diff+diff_frame
    i=i+1
#差分を正規化
diff = (diff/diff.max())*255
diff = diff.astype(np.uint8)
#cv2.imwrite(os.path.join(self.__DIF_DIR, self.VIDEO_NAME + self.__IMG_EXT), self.diff) 

#差分を二値化
ret, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)

#カメラの枠外を除去
mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
cv2.circle(mask, center=(img.shape[2] // 2, img.shape[1] // 2), radius=img.shape[1]//2, color=255, thickness=-1)
diff[mask == 0] = 0
cv2.imwrite('/home/pi/marker/diff/dif-asdfg.png',diff)

t3=time.time()

#窓関数をかける
img = img.transpose(1, 2, 0)
masked_img = img[diff == 255]   
print(masked_img.shape)    
img = img.transpose(2, 0, 1)
masked_img = masked_img * __win     
acf = 1/(__win.sum()/ __N)
masked_img = masked_img.transpose(1,0)

#fft処理
t4=time.time()
fft_signal = np.fft.fft(masked_img, axis=0)
t5=time.time()

#bpfに通す
filter_ = np.logical_and(__freq < 5.3, __freq > 4.8)
fft_signal_filtered = fft_signal[filter_]
t6=time.time()

#正規化
fft_signal_filtered_amp = acf * np.abs(fft_signal_filtered)
fft_signal_filtered_amp = fft_signal_filtered_amp / (__N / 2)
#print(fft_signal_filtered_amp.shape)
fft_signal_filtered_amp[0, :] /= 2
t7=time.time()

#
output = np.zeros_like(img[:1])
output = output.transpose(1,2,0)
output[diff == 255] = fft_signal_filtered_amp.transpose(1,0)
#print(fft_signal_filtered_amp)
output = output.transpose(2,0,1)
output = np.max(output, axis=0).astype(np.uint8)

#2値化
output[output.max() > output] = 0
output[output.max() <= output] = 255

#出力画像と切り抜き画像の重ね合わせ
output = cv2.resize(output, (3280, 2464))
cv2.imwrite('/home/pi/marker/output/dif-asdfg.png',output)
t8=time.time()
T=t8-t1
print(f"経過時間：{T}")
T=t2-t1
print(f"フレーム分割：{T}")
T=t3-t2
print(f"差分：{T}")
T=t4-t3
print(f"窓関数：{T}")
T=t5-t4
print(f"FFT：{T}")
T=t6-t5
print(f"bpf：{T}")
T=t7-t6
print(f"正規化：{T}")
T=t8-t7
print(f"画像生成：{T}")