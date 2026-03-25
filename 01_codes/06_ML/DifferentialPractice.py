# chap6. 머신러닝

'''
06-04. 자동 미분과 선형 회귀 실습
'''

import os

# TensorFlow startup 로그를 줄여서 예제 출력이 덜 시끄럽게 보이도록 설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


# 자동 미분
w = tf.Variable(2.)

def fun1(w):
    y = w**2
    z = 2*y + 5
    return z

with tf.GradientTape() as type:
    z = fun1(w)

gradients = type.gradient(z, [w])
print(gradients)
# [<tf.Tensor: shape=(), dtype=float32, numpy=8.0>]


# 자동 미분으로 선형 회귀 구현
# 학습시킬 파라미터
w = tf.Variable(4.0)
b = tf.Variable(1.0)

@tf.function
def hypothesis(x):
    return w*x + b

x_test = [3.5, 5.5, 6]
print(hypothesis(x_test).numpy())
# [15. 23. 25.]


# MSE를 손실함수로 정의
@tf.function
def mse_loss(y_pred, y):
  # 두 개의 차이값을 제곱을 해서 평균을 취한다.
  return tf.reduce_mean(tf.square(y_pred - y))

x = [1, 2, 3, 4, 5, 6, 7, 8, 9] # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95] # 각 공부하는 시간에 맵핑되는 성적

# SGD 방식 경사하강법 사용, 학습률 0.01
optimizer = tf.optimizers.SGD(0.01)

# 300번 GD 수행
for i in range(301):
  with tf.GradientTape() as tape:
    # 현재 파라미터에 기반한 입력 x에 대한 예측값을 y_pred
    y_pred = hypothesis(x)

    # 평균 제곱 오차를 계산
    cost = mse_loss(y_pred, y)

  # 손실 함수에 대한 파라미터의 미분값 계산
  gradients = tape.gradient(cost, [w, b])

  # 파라미터 업데이트
  optimizer.apply_gradients(zip(gradients, [w, b]))

  if i % 10 == 0:
    print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))
'''
epoch :   0 | w의 값 : 8.2133 | b의 값 : 1.664 | cost : 1402.555542
epoch :  10 | w의 값 : 10.4971 | b의 값 : 1.977 | cost : 1.351182
epoch :  20 | w의 값 : 10.5047 | b의 값 :  1.93 | cost : 1.328165
epoch :  30 | w의 값 : 10.5119 | b의 값 : 1.884 | cost : 1.306967
epoch :  40 | w의 값 : 10.5188 | b의 값 : 1.841 | cost : 1.287436
epoch :  50 | w의 값 : 10.5254 | b의 값 : 1.799 | cost : 1.269459
epoch :  60 | w의 값 : 10.5318 | b의 값 : 1.759 | cost : 1.252898
epoch :  70 | w의 값 : 10.5379 | b의 값 : 1.721 | cost : 1.237644
epoch :  80 | w의 값 : 10.5438 | b의 값 : 1.684 | cost : 1.223598
epoch :  90 | w의 값 : 10.5494 | b의 값 : 1.648 | cost : 1.210658
epoch : 100 | w의 값 : 10.5548 | b의 값 : 1.614 | cost : 1.198740
epoch : 110 | w의 값 : 10.5600 | b의 값 : 1.582 | cost : 1.187767
epoch : 120 | w의 값 : 10.5650 | b의 값 :  1.55 | cost : 1.177665
epoch : 130 | w의 값 : 10.5697 | b의 값 :  1.52 | cost : 1.168354
epoch : 140 | w의 값 : 10.5743 | b의 값 : 1.492 | cost : 1.159782
epoch : 150 | w의 값 : 10.5787 | b의 값 : 1.464 | cost : 1.151890
epoch : 160 | w의 값 : 10.5829 | b의 값 : 1.437 | cost : 1.144619
epoch : 170 | w의 값 : 10.5870 | b의 값 : 1.412 | cost : 1.137924
epoch : 180 | w의 값 : 10.5909 | b의 값 : 1.387 | cost : 1.131752
epoch : 190 | w의 값 : 10.5946 | b의 값 : 1.364 | cost : 1.126073
epoch : 200 | w의 값 : 10.5982 | b의 값 : 1.341 | cost : 1.120843
epoch : 210 | w의 값 : 10.6016 | b의 값 :  1.32 | cost : 1.116026
epoch : 220 | w의 값 : 10.6049 | b의 값 : 1.299 | cost : 1.111589
epoch : 230 | w의 값 : 10.6081 | b의 값 : 1.279 | cost : 1.107504
epoch : 240 | w의 값 : 10.6111 | b의 값 :  1.26 | cost : 1.103736
epoch : 250 | w의 값 : 10.6140 | b의 값 : 1.242 | cost : 1.100273
epoch : 260 | w의 값 : 10.6168 | b의 값 : 1.224 | cost : 1.097082
epoch : 270 | w의 값 : 10.6195 | b의 값 : 1.207 | cost : 1.094143
epoch : 280 | w의 값 : 10.6221 | b의 값 : 1.191 | cost : 1.091434
epoch : 290 | w의 값 : 10.6245 | b의 값 : 1.176 | cost : 1.088940
epoch : 300 | w의 값 : 10.6269 | b의 값 : 1.161 | cost : 1.086645
'''

# 학습된 w, b로 다시 x_test에 대해 hypothesis 함수 적용
x_test = [3.5, 5, 5.5, 6]
print(hypothesis(x_test).numpy())
# [38.35479  54.295143 59.608593 64.92204 ]