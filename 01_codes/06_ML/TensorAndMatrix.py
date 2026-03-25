# 06-08. 벡터와 행렬 연산
'''
벡터
- 크기와 방향을 가진 값
- 파이썬에서는 1차원 배열 혹은 리스트로 표현

행렬 
- 행과 열을 가지는 2차원 형상
- 파이썬에서는 2차원 배열

3차원부터 텐서라고 부름
'''

import numpy as np

# 0차원 텐서 - 스칼라
d = np.array(5)
print('텐서의 차원 :',d.ndim)
print('텐서의 크기(shape) :',d.shape)

# 1차원 텐서 - 벡터(1D Tensor)
d = np.array([1, 2, 3, 4])
print('텐서의 차원 :',d.ndim)
print('텐서의 크기(shape) :',d.shape)

# 2차원 텐서 - 행렬(2D Tensor)
# 3행 4열의 행렬
d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('텐서의 차원 :',d.ndim)
print('텐서의 크기(shape) :',d.shape)

# 3차원 텐서
d = np.array([
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [19, 20, 21, 22, 23], [23, 24, 25, 26, 27]]
            ])
print('텐서의 차원 :',d.ndim)
print('텐서의 크기(shape) :',d.shape)

'''
텐서의 차원 : 0
텐서의 크기(shape) : ()
텐서의 차원 : 1
텐서의 크기(shape) : (4,)
텐서의 차원 : 2
텐서의 크기(shape) : (3, 4)
텐서의 차원 : 3
텐서의 크기(shape) : (2, 3, 5)

NLP에서 주로 3차원 텐서를 자주 본다.
보통 (sample_nums, time_steps, dim)으로 구성된다.
배치 적용시 (batch_size, time_steps, dim)으로 볼수도 있다.

'''

# 벡터 내적과 행렬곱
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
print('두 벡터의 내적 :',np.dot(A, B))
# 두 벡터의 내적 : 32

A = np.array([[1, 3],[2, 4]])
B = np.array([[5, 7],[6, 8]])
print('두 행렬의 행렬곱 :')
print(np.matmul(A, B))
# 두 행렬의 행렬곱 :
# [[23 31]
#  [34 46]]
