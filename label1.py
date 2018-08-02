import cv2
import numpy as np
import random
import sys

# 画像の読み込み
img = cv2.imread('sample.jpg')

# グレースケール化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 大津の二値化
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# 白黒反転
gray = cv2.bitwise_not(gray)

# ラベリング処理(簡易版)
n, label = cv2.connectedComponents(gray)

# ラベリング結果書き出し準備
color_src = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
height, width = gray.shape[:2]
colors = []

for i in range(1, n + 1):
    colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

# ラベリング結果を表示
# 各オブジェクトをランダム色でペイント
for y in range(0, height):
    for x in range(0, width):
        if label[y, x] > 0:
            color_src[y, x] = colors[label[y, x]]
        else:
            color_src[y, x] = [0, 0, 0]

# オブジェクトの総数を黄文字で表示
cv2.putText(color_src, str(n - 1), (70, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

# 画像の保存
cv2.imwrite('sample_label1.jpg', color_src)
