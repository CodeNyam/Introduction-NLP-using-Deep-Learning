'''
Part 2. 텍스트 전처리
'''

'''
02-07. 패딩
문서, 문장의 길이를 맞추어주는 작업
'''

# Numpy 사용
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
# 예시 데이터
preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

# 단어 집합 만들기 & 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)
# 결과
# [[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]

# 최대 길이 확인
max_len = max(len(item) for item in encoded)
print('최대 길이 :',max_len)

# 최대 길이에 맞추어서 인코딩
# 인코딩 길이가 짧을 경우 제로 패딩
for sentence in encoded:
    while len(sentence) < max_len:
        sentence.append(0)

padded_np = np.array(encoded)
print(padded_np)
# 결과
# [[ 1  5  0  0  0  0  0]
#  [ 1  8  5  0  0  0  0]
#  [ 1  3  5  0  0  0  0]
#  [ 9  2  0  0  0  0  0]
#  [ 2  4  3  2  0  0  0]
#  [ 3  2  0  0  0  0  0]
#  [ 1  4  6  0  0  0  0]
#  [ 1  4  6  0  0  0  0]
#  [ 1  4  2  0  0  0  0]
#  [ 7  7  3  2 10  1 11]
#  [ 1 12  3 13  0  0  0]]


# 캐라스 사용
from tensorflow.keras.preprocessing.sequence import pad_sequences

encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
padded = pad_sequences(encoded, padding='post') # 기본값은 뒤부터 제로 패딩, padding='post' 변경시 앞에서 제로 패딩
print(padded)
# 결과
# [[ 1  5  0  0  0  0  0]
#  [ 1  8  5  0  0  0  0]
#  [ 1  3  5  0  0  0  0]
#  [ 9  2  0  0  0  0  0]
#  [ 2  4  3  2  0  0  0]
#  [ 3  2  0  0  0  0  0]
#  [ 1  4  6  0  0  0  0]
#  [ 1  4  6  0  0  0  0]
#  [ 1  4  2  0  0  0  0]
#  [ 7  7  3  2 10  1 11]
#  [ 1 12  3 13  0  0  0]]

padded = pad_sequences(encoded, padding='post', maxlen=5) # maxlen=5 : 최대 길이를 5로 제한
print(padded)
# 결과
# [[ 1  5  0  0  0]
#  [ 1  8  5  0  0]
#  [ 1  3  5  0  0]
#  [ 9  2  0  0  0]
#  [ 2  4  3  2  0]
#  [ 3  2  0  0  0]
#  [ 1  4  6  0  0]
#  [ 1  4  6  0  0]
#  [ 1  4  2  0  0]
#  [ 7  7  3  2 10]
#  [ 1 12  3 13  0]]

# 다른 값으로 패딩하기
padded = pad_sequences(encoded, padding='post', value=1)
print(padded)
# 결과
# [[ 1  5  1  1  1  1  1]
#  [ 1  8  5  1  1  1  1]
#  [ 1  3  5  1  1  1  1]
#  [ 9  2  1  1  1  1  1]
#  [ 2  4  3  2  1  1  1]
#  [ 3  2  1  1  1  1  1]
#  [ 1  4  6  1  1  1  1]
#  [ 1  4  6  1  1  1  1]
#  [ 1  4  2  1  1  1  1]
#  [ 7  7  3  2 10  1 11]
#  [ 1 12  3 13  1  1  1]]

