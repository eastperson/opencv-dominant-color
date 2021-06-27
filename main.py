import cv2
import numpy as np
from skimage import io

img = io.imread('image/4.png')[:, :, :-1]

# 높이, 너비, 채널
print('img :',img.shape)

# 각 색채 채널의 평균을 계산
average = img.mean(axis=0).mean(axis=0)

print('average :',average)

#  k-means clustering를 적용하여 가장 대표적인 이미지 색상의 팔레트를 만든다.
# reshape 메써드의 첫번째 파라미터는 채널의 새로운 개수이고 두번째 파라미터는 행(rows)의 새로운 개수이다.
# 열(columns)의 개수는 자동으로 계산
pixels = np.float32(img.reshape(-1, 3))
print('pixels :',pixels)


n_colors = 8
print('n_colors :',n_colors)
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
# cv2.TERM_CRITERIA_EPS : epsilon으로 주어진 정확도를 만족하면 반복을 멈춘다.
# cv2.TERM_CRITERIA_MAX_ITER : 정확도와 상관없이 미리 정해진 반복 횟수를 다 채우면 알고리즘을 멈춘다.
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER : 둘 중 하나의 조건에 해당하면 멈춘다. 200회가 최대 반복횟수이고 .1이 정확도이다.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
print('criteria :',criteria)

# cv2.KMEANS_RANDOM_CENTERS
# flags : 이 flag는 어떻게 초기 중심값을 정할지에 대한 것이다.
# flags : 초기 중앙값 설정 방법 (kmeans에서 초기 k개의 중앙값을 어떻게 설정하느냐에 따라 결과 차이가 존재합니다.)

# flag 1) cv2.KMEANS_RANDOM_CENTERS = 초기 중앙값을 랜덤 지정 (가장 기본적인 방법)
# flag 2) cv2.KMEANS_PP_CENTERS = K-means++ 알고리즘을 이용하여 지정 (시간이 소요되지만, 랜덤 지정보다 정확도가 좋음)
# flag 3) cv2.KMEANS_USE_INITIAL_LABELS = 사용자가 k개의 중앙값 지정
flags = cv2.KMEANS_RANDOM_CENTERS
print('flags :',flags)

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
print('ret :',_,', labels :',labels,', palette :',palette)

# np.unique() 메소드를 사용하면 numpy 배열 내 고유한 원소(unique elements)의 집합을 찾을 수 있습니다.
# unique(x) 배열 x에서 중복된 원소를 제거한 후 정렬(sorted)하여 반환
# np.unique(arr) [Out] array(['a', 'b', 'c'], dtype='<U1')
# np.unique(arr, return_inverse=True) [Out] (array(['a', 'b', 'c'], dtype='<U1'), array([0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 1, 0, 0, 0, 2]))
_, counts = np.unique(labels, return_counts=True)
print('ret : ', _,', counts :',counts)

# np.argmax()는 최대값을 가진 위치를 반환하는 함수
# np.argmax(data) 가 2라면 2번째 위치에 최대값이 존재
dominant = palette[np.argmax(counts)]
print('np.argmax(counts) :',np.argmax(counts))
print('dominant :',dominant)

import matplotlib.pyplot as plt

# 전달인자로 전달한 dtype과 모양(행,렬)으로 배열을 생성하고 모든 내용을 1로 초기화하여 ndarray를 반환
# def ones(shape, dtype=None, order='C', *, like=None)
# img의 행렬과 같은 크기의 화면을 1로 반환

# np.uint8은 양수만 표현가능, 2^8개수까지 표현가능, 0~255 -> 소수점자리를 버리고 정수로 반환
# 넘파이 행렬을 average로 곱하고 저장한다.
avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average)

# np.argsort()는 넘파이 배열의 원소를 오름차순으로 정렬하는 메소드
# 슬라이싱으로 내림차순으로 변환
indices = np.argsort(counts)[::-1]
print('indices :',indices)

# np.hstack()은 행 방향으로 합쳐주는 메서드
# ex) A1 : array([1,2,3]), B1 : array([4,5,6])
# np.hstack(A1,B1) -> array([1,2,3,4,5,6])
# np.vstack(A1,B1) -> array([1,2,3]
#                           ,[4,5,6])
# np.hstack([[0], counts[indices] / counts.sum()]

# np.cumsum : 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수
# def cumsum(a, axis=None, dtype=None, out=None)
# cumsum(a) : 각 원소들의 누적 합을 표시함. 각 row와 column의 구분은 없어지고, 순서대로 sum을 함.
# cumsum(a, dtype = float) : 결과 값의 변수 type을 설정하면서 누적 sum을 함.
# cumsum(a, axis = 0) : axis = 0은 같은 column 끼리의 누적 합을 함.
# cumsum(a, axis = 1) : axis = 1은 같은 row끼리의 누적 합을 함

# freqs = np.cumsum(np.hstack([[0], counts[indices] / counts.sum()]))
# 0부터 indices를 sum으로 나눈 값들을 누적으로 합한 합을 보여준다.
# 사각형의 비중을 나타낸다.
freqs = np.cumsum(np.hstack([[0], counts[indices] / counts.sum()]))
print('freqs :',freqs)

# 높이에 freqs를 곱해 row를 생성한다.
rows = np.int_(img.shape[0] * freqs)
print('rows :',rows)

dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
result = []
for i in range(len(rows) - 1):
    dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
    li = []
    li.append(np.uint8(palette[indices[i]])[0])
    li.append(np.uint8(palette[indices[i]])[1])
    li.append(np.uint8(palette[indices[i]])[2])
    result.append(li)
    print('rows[',i,']:rows[',i,' + 1] :',rows[i])
    print('palette[indices[',i,'] :',np.uint8(palette[indices[i]]))

print('result :',result)

# print('dom_pathch : ',dom_patch.imag)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
ax0.imshow(avg_patch)
ax0.set_title('Average color')
ax0.axis('off')
ax1.imshow(dom_patch)
ax1.set_title('Dominant colors')
ax1.axis('off')
plt.show()