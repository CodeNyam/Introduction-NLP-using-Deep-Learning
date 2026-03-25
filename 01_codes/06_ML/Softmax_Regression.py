# 06-09. 소프트맥스 회귀
'''
앞서 이진 분류를 위한 로지스틱 회귀법으로 시그모이드를 도입했다.
이제는 다중 분류에서 로지스틱 회귀를 적용하기 위해 소프트맥스 회귀법을 배운다.
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/06.%20Machine%20Learning/dataset/Iris.csv", filename="Iris.csv")

data = pd.read_csv('Iris.csv', encoding='latin1')

print('샘플의 개수 :', len(data))
print(data[:5])
'''
샘플의 개수 : 150
   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa
'''


# 중복을 허용하지 않고, 있는 데이터의 모든 종류를 출력
print("품종 종류:", data["Species"].unique(), sep="\n")
'''
품종 종류:
<StringArray>
['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
Length: 3, dtype: str
'''


sns.set_theme(style="ticks")
g = sns.pairplot(data, hue="Species", palette="husl")
g.savefig("softmax_pairplot.png", dpi=150)
plt.close(g.fig)

# 각 종과 특성에 대한 연관 관계
plt.figure()
sns.barplot(x='Species', y='SepalWidthCm', data=data, errorbar=None)
plt.tight_layout()
plt.savefig("softmax_barplot.png", dpi=150)
plt.close()

plt.figure()
data['Species'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.savefig("softmax_species_counts.png", dpi=150)
plt.close()

# Iris-virginica는 0, Iris-setosa는 1, Iris-versicolor는 2가 됨.
data['Species'] = data['Species'].replace(['Iris-virginica','Iris-setosa','Iris-versicolor'],[0,1,2])
plt.figure()
data['Species'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.savefig("softmax_species_encoded_counts.png", dpi=150)
plt.close()


# X 데이터. 특성은 총 4개.
data_X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

# Y 데이터. 예측 대상.
data_y = data['Species'].values

print(data_X[:5])
print(data_y[:5])


# 검증
# 훈련 데이터와 테스트 데이터를 8:2로 나눈다.
(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.8, random_state=1)

# 원-핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train[:5])
print(y_test[:5])



# 소프트 맥스 회귀
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test))


epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.tight_layout()
plt.savefig('softmax_reggression.png', dpi=150)
plt.close()


print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
# 테스트 정확도: 0.9667