# 06-07. 다중 입력에 대한 실습
# 독립 변수 x가 2개 이상인 경우 학습

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


# 중간 고사, 기말 고사, 가산점 점수 구조로 입력 데이터 구성
X = np.array([[70,85,11], [71,89,18], [50,80,20], [99,20,10], [50,10,10]]) 
y = np.array([73, 82 ,72, 57, 34]) # 최종 성적

model = Sequential()
model.add(Dense(1, input_dim=3, activation='linear'))

sgd = optimizers.SGD(learning_rate=0.0001)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
model.fit(X, y, epochs=2000)


# 예측값 출력
print(model.predict(X))
# [[72.921745]
#  [81.97953 ]
#  [72.10089 ]
#  [57.118626]
#  [33.802853]]


X_test = np.array([[20,99,10], [40,50,20]])
print(model.predict(X_test))
# [[57.908222]
#  [56.052586]]
