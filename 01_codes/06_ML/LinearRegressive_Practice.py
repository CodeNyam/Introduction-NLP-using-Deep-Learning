import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float).reshape(-1, 1)  # 공부하는 시간
y = np.array([11, 22, 33, 44, 53, 66, 77, 87, 95], dtype=float)  # 각 공부하는 시간에 맵핑되는 성적

model = Sequential()

# 출력 y의 차원은 1. 입력 x의 차원(input_dim)은 1
# 선형 회귀이므로 activation은 'linear'
model.add(Dense(1, input_dim=1, activation='linear'))

# sgd는 경사 하강법을 의미. 학습률(learning rate)은 0.01.
sgd = optimizers.SGD(learning_rate=0.01)

# 손실 함수(Loss function)은 평균제곱오차 mse를 사용합니다.
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

# 주어진 x와 y데이터에 대해서 오차를 최소화하는 작업을 300번 시도합니다.
model.fit(x, y, epochs=300)
# 결과
'''
...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - loss: 1.0881 - mse: 1.0881
Epoch 270/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - loss: 1.0878 - mse: 1.0878
Epoch 271/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 1.0876 - mse: 1.0876
Epoch 272/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step - loss: 1.0874 - mse: 1.0874
Epoch 273/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 1.0872 - mse: 1.0872
Epoch 274/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - loss: 1.0869 - mse: 1.0869
Epoch 275/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - loss: 1.0867 - mse: 1.0867
Epoch 276/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - loss: 1.0865 - mse: 1.0865
Epoch 277/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 1.0863 - mse: 1.0863
Epoch 278/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step - loss: 1.0861 - mse: 1.0861
Epoch 279/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step - loss: 1.0858 - mse: 1.0858
Epoch 280/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 1.0856 - mse: 1.0856
Epoch 281/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - loss: 1.0854 - mse: 1.0854
Epoch 282/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 1.0852 - mse: 1.0852
Epoch 283/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - loss: 1.0850 - mse: 1.0850
Epoch 284/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - loss: 1.0848 - mse: 1.0848
Epoch 285/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 1.0846 - mse: 1.0846
Epoch 286/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - loss: 1.0844 - mse: 1.0844
Epoch 287/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step - loss: 1.0842 - mse: 1.0842
Epoch 288/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step - loss: 1.0840 - mse: 1.0840
Epoch 289/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - loss: 1.0838 - mse: 1.0838
Epoch 290/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - loss: 1.0836 - mse: 1.0836
Epoch 291/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - loss: 1.0834 - mse: 1.0834
Epoch 292/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 1.0832 - mse: 1.0832
Epoch 293/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 1.0830 - mse: 1.0830
Epoch 294/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 1.0828 - mse: 1.0828
Epoch 295/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 1.0826 - mse: 1.0826
Epoch 296/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - loss: 1.0824 - mse: 1.0824
Epoch 297/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 1.0823 - mse: 1.0823
Epoch 298/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 1.0821 - mse: 1.0821
Epoch 299/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 1.0819 - mse: 1.0819
Epoch 300/300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - loss: 1.0817 - mse: 1.0817
'''

plt.plot(x, model.predict(x), 'b', x, y, 'k')

print(model.predict([9.5]))
