# 06. ML
# 06-05. 로지스틱 회귀

'''
이진 분류 문제를 풀기위한 알고리즘으로 로지스틱 회귀를 사용할 수 있다.
ex. 시험 점수가 70점 이상일때만 합격(1), 그 미만은 불합격(0)으로 표기
- 시험 점수를 x, 여부를 y로 설정하면, 70점 미만은 y = 0, 이상은 y = 1에 할당
=> x와 y의 관계가 S자 형태로 표현됨
이 경우 일반 선형 회귀 방식은 사용할 수 없음.

로지스틱 회귀법 도입을 위해 아래를 가정
- 예측값을 0과 1이 아닌 [0, 1] 범위에 존재한다고 가정
    - 따라서 예측값이 0.5보다 작을 경우 0, 클 경우 1로 분류할 수 있음.

대표적으로 로지스틱 회귀에 시그모이드 함수가 적용
'''
import os

# TensorFlow startup 로그를 줄여서 예제 출력이 덜 시끄럽게 보이도록 설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.tight_layout()
plt.savefig('sigmoid_function.png', dpi=150)
plt.close()


# 가중치와 편향 변경에 따른 시그모이드 함수
def modified_weight_sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--') # w의 값이 0.5일때
plt.plot(x, y2, 'g') # w의 값이 1일때
plt.plot(x, y3, 'b', linestyle='--') # w의 값이 2일때
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.tight_layout()
plt.savefig('modified_weight_sigmoid.png', dpi=150)
plt.close()
# w에 따라 그래프 경사가 변함.
# w 크기가 클수록 경사가 커짐


def modified_bias_sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--') # x + 0.5
plt.plot(x, y2, 'g') # x + 1
plt.plot(x, y3, 'b', linestyle='--') # x + 1.5
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.tight_layout()
plt.savefig('modified_bias_sigmoid.png', dpi=150)
plt.close()
# b 값에 따라서 그래프가 x축 방향에서 이동


# 케라스로 구현한 로지스틱 회귀
x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50], dtype=float).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 숫자 10부터 1

model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(x, y, epochs=200)

plt.plot(x.squeeze(), model.predict(x, verbose=0), 'b', x.squeeze(), y, 'k.')
plt.tight_layout()
plt.savefig('logistic_regression.png', dpi=150)
plt.close()

print(model.predict(np.array([[1], [2], [3], [4], [4.5]]), verbose=0))
print(model.predict(np.array([[11], [21], [31], [41], [500]]), verbose=0))
'''
[[0.50159216]
 [0.5570799 ]
 [0.61117876]
 [0.6626712 ]
 [0.68712145]]
[[0.9034251 ]
 [0.9886304 ]
 [0.99876434]
 [0.9998669 ]
 [1.        ]]
'''