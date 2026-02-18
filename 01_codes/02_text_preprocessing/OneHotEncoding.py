'''
Part 2. 텍스트 전처리
'''

'''
02-08. 원-핫 인코딩
해당 위치에 토큰의 존재 유무를 표현하는 인코딩 벡터
'''

# Okt 형태소 분석기로 문장 토큰화
from konlpy.tag import Okt  

okt = Okt()  
tokens = okt.morphs("나는 자연어 처리를 배운다")  
print(tokens)
# ['나', '는', '자연어', '처리', '를', '배운다']

# 각 토큰에 고유 정수 부여
word_to_index = {word : index for index, word in enumerate(tokens)}
print('단어 집합 :',word_to_index)
# 단어 집합 : {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}


# 원-핫 인코딩 함수
def one_hot_encoding(word, word_to_index):
  one_hot_vector = [0]*(len(word_to_index))
  index = word_to_index[word]
  one_hot_vector[index] = 1
  return one_hot_vector

print(one_hot_encoding("자연어", word_to_index))
# 결과
# [0, 0, 1, 0, 0, 0]


# 케라스 사용
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print('단어 집합 :',tokenizer.word_index)
# 단어 집합 : {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}

sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)
# [2, 5, 1, 6, 3, 7]

# 케라스의 원-핫 인코딩 기능
one_hot = to_categorical(encoded)
print(one_hot)
# [[0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]

'''
원-핫 인코딩의 한계
1. 벡터 차원이 늘어날수록 비효율적, 임베딩 벡터가 차지하는 저장 공간만 늘어남
2. 단어 유사도 표현 불가
ex. 개와 고양이의 원-핫 인코딩 벡터가 각각 [0,0,1], [0,1,0]인 상황
이때 늑대와 삵이 각각 [1,0,1], [1,1,0]이라고 하면, 늑대-개, 삵-고양이와 유사도를 표현할 수 없음
원-핫 인코딩은 이런 단점 때문에 검색 시스템에서 사용하기 불가능
이를 위해 단어의 잠재 의미를 반영해 다차원 공간에 벡터화하는 기법이 존재
1. LSA(잠재 의미 분석), HAL - 카운트 기반 벡터화
2. NNLM, RNNLM, Word2Vec, FastText - 예측 기반 벡터화

