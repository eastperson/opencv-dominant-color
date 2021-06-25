import cv2
import numpy as np
from skimage import io

img = io.imread('https://shopping-phinf.pstatic.net/main_2191829/21918294654.20200428094627.jpg?type=f640')[:, :, :-1]

print('img :',img.shape)

# 각 색채 채널의 평균을 계산
average = img.mean(axis=0).mean(axis=0)

#  k-means clustering를 적용하여 가장 대표적인 이미지 색상의 팔레트를 만든다.
pixels = np.float32(img.shape)

n_colors = 5
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
# cv2.TERM_CRITERIA_EPS : epsilon으로 주어진 정확도를 만족하면 반복을 멈춘다.
# cv2.TERM_CRITERIA_MAX_ITER : 정확도와 상관없이 미리 정해진 반복 횟수를 다 채우면 알고리즘을 멈춘다.
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER : 둘 중 하나의 조건에 해당하면 멈춘다. 200회가 최대 반복횟수이고 .1이 정확도이다.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)

# cv2.KMEANS_RANDOM_CENTERS
# flags : 이 flag는 어떻게 초기 중심값을 정할지에 대한 것이다.
# flags : 초기 중앙값 설정 방법 (kmeans에서 초기 k개의 중앙값을 어떻게 설정하느냐에 따라 결과 차이가 존재합니다.)

# flag 1) cv2.KMEANS_RANDOM_CENTERS = 초기 중앙값을 랜덤 지정 (가장 기본적인 방법)
# flag 2) cv2.KMEANS_PP_CENTERS = K-means++ 알고리즘을 이용하여 지정 (시간이 소요되지만, 랜덤 지정보다 정확도가 좋음)
# flag 3) cv2.KMEANS_USE_INITIAL_LABELS = 사용자가 k개의 중앙값 지정
flags = cv2.KMEANS_RANDOM_CENTERS

# cv2.kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None)
# data: 학습 데이터 행렬. numpy.ndarray. shape=(N, d), dtype=numpy.float32.
# K: 군집 개수
# bestLabels: 각 샘플의 군집 번호 행렬. numpy.ndarray. shape=(N, 1),dtype=np.int32.
# criteria: 종료 기준. (type, maxCount, epsilon) 튜플.
# attempts: 다른 초기 레이블을 이용해 반복 실행할 횟수.
# flags: 초기 중앙 설정 방법. cv2.KMEANS_RANDOM_CENTERS, cv2.KMEANS_PP_CENTERS, cv2.KMEANS_USE_INITIAL_LABELS 중 하나.
# centers: 군집 중심을 나타내는 행렬. np.ndarray. shape=(N, d), dtype=np.float32.

# K는 데이터를 분류할 클러스터의 수이다. 이 예제에서는 n_colors가 클러스터의 수인 것이다.
# attempts는 알고리즘 실행 횟수이다.
# 따라서 attempts의 독립적인 실행으로 분류를 한 후 최적의 밀집도 결과를 리턴하게 되는 것이다.

# ret, labels, centers = cv2.kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None)는 3개의 output 변수를 리턴한다.
# ret : 밀집도(compactness)이다. 각 클러스터의 중심으로부터 거리제곱의 합이다.
# labels : 각 점이 클러스터에 속하게 되었는지 표시한다.
# centers : 클러스터의 중심 좌표를 리턴한다. K개의 데이터를 가진다.
_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

# np.unique() 메소드를 사용하면 numpy 배열 내 고유한 원소(unique elements)의 집합을 찾을 수 있습니다.
# np.unique(arr) [Out] array(['a', 'b', 'c'], dtype='<U1')
# np.unique(arr, return_inverse=True) [Out] (array(['a', 'b', 'c'], dtype='<U1'), array([0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 1, 0, 0, 0, 2]))

_, counts = np.unique(labels, return_counts=True)

dominant = palette[np.argmax(counts)]

import matplotlib.pyplot as plt

avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average)

indices = np.argsort(counts)[::-1]
freqs = np.cumsum(np.hstack([[0], counts[indices] / counts.sum()]))
rows = np.int_(img.shape[0] * freqs)

dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
for i in range(len(rows) - 1):
    dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
ax0.imshow(avg_patch)
ax0.set_title('Average color')
ax0.axis('off')
ax1.imshow(dom_patch)
ax1.set_title('Dominant colors')
ax1.axis('off')
plt.show()