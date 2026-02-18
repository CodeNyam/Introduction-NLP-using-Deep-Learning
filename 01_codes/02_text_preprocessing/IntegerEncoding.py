'''
Part 2. 텍스트 전처리
'''

'''
02-06. 정수 인코딩
컴퓨터는 숫자를 처리함
그 이전에 각 단어에 고유한 정수값을 매핑시키는 전처리 작업
해당 작업은 원-핫 인코딩과 관련
'''
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

raw_sentences = f"""
    A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is False.
    A piece of paper is white. a paper is good. a paper is great. he Knew A Secret! The Secret He Kept is False.
    A coin is a coin. a coin is good. a coin is great. he Knew A Secret! The Secret He Kept is False.
    """

# 1. 딕셔너리 사용
tokenized_sentences = sent_tokenize(raw_sentences)
print(tokenized_sentences)
# 결과
# ['\n    A barber is a person.', 'a barber is good person.', 'a barber is huge person.', 'he Knew A Secret!', 'The Secret He Kept is False.', 'A piece of paper is white.', 'a paper is good.', 'a paper is great.', 'he Knew A Secret!', 'The Secret He Kept is False.', 'A coin is a coin.', 'a coin is good.', 'a coin is great.', 'he Knew A Secret!', 'The Secret He Kept is False.']

vocab = {}
preprocessed_sentence = []
stopwords_set = set(stopwords.words('english'))

for sentence in tokenized_sentences:
    tokenized_sentence_words = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence_words:
        word = word.lower() # 모든 단어를 소문자화해 단어 개수를 줄임
        if word not in stopwords_set:
            if len(word) > 2:
                result.append(word)
                
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1

    preprocessed_sentence.append(result)
    
print(preprocessed_sentence)
# 결과
# ['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'false'], ['piece', 'paper', 'white'], ['paper', 'good'], ['paper', 'great'], ['knew', 'secret'], ['secret', 'kept', 'false'], ['coin', 'coin'], ['coin', 'good'], ['coin', 'great'], ['knew', 'secret'], ['secret', 'kept', 'false']]

print(f"단어 집합(Vocabulary)의 크기: {len(vocab)}")
print(f"단어 집합(Vocabulary): {vocab}")
# 결과
# 단어 집합(Vocabulary)의 크기: 13
# 단어 집합(Vocabulary): {'barber': 3, 'person': 3, 'good': 3, 'huge': 1, 'knew': 3, 'secret': 6, 'kept': 3, 'false': 3, 'piece': 1, 'paper': 3, 'white': 1, 'great': 2, 'coin': 4}

print(vocab["barber"])
# 결과
# 3

# 빈도수가 높은 순서대로 나열 후 정수값 라벨링
vocab_sorted = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
print(vocab_sorted)
# 결과
# [('secret', 6), ('barber', 3), ('good', 3), ('knew', 3), ('kept', 3), ('false', 3), ('piece', 1), ('paper', 3), ('white', 1), ('great', 2), ('coin', 4), ('person', 3), ('huge', 1)]

word_to_index = {}
i = 0
for word, index in vocab_sorted:
    if index > 0:
        i += 1
        word_to_index[word] = i
print(word_to_index)
# 결과
# {'secret': 1, 'coin': 2, 'barber': 3, 'person': 4, 'good': 5, 'knew': 6, 'kept': 7, 'false': 8, 'paper': 9, 'great': 10, 'huge': 11, 'piece': 12, 'white': 13}

# 빈도수가 높은 5개 단어만 사용
# 단어 길이가 5 초과이면 제거
vocab_size = 5
word_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]

for word in word_frequency:
    del word_to_index[word]
print(word_to_index)
# 결과
# {'secret': 1, 'coin': 2, 'barber': 3, 'person': 4, 'good': 5}

# Out-Of-Vocabulary(OOV)인 경우
word_to_index["OOV"] = len(word_to_index) + 1
print(word_to_index)
# 결과
# {'secret': 1, 'coin': 2, 'barber': 3, 'person': 4, 'good': 5, 'OOV': 6}

# 정수 인코딩
encoded_sentences = []
for sentence in preprocessed_sentence:
    encoded_sentence = []
    for word in sentence:
        try: # 단어 집합에 있는 단어이면 해당 정수값으로 인코딩
            encoded_sentence.append(word_to_index[word])
        except KeyError: # 단어 집합에 없는 경우 OOV로 인코딩
            encoded_sentence.append(word_to_index["OOV"])
    encoded_sentences.append(encoded_sentence)
print(encoded_sentences)
# 결과
# [[3, 4], [3, 5, 4], [3, 6, 4], [6, 1], [1, 6, 6], [6, 6, 6], [6, 5], [6, 6], [6, 1], [1, 6, 6], [2, 2], [2, 5], [2, 6], [6, 1], [1, 6, 6]]


# Counter 사용하기
from collections import Counter

all_words_list = sum(preprocessed_sentence, [])
print(all_words_list)
# 결과
# ['barber', 'person', 'barber', 'good', 'person', 'barber', 'huge', 'person', 'knew', 'secret', 'secret', 'kept', 'false', 'piece', 'paper', 'white', 'paper', 'good', 'paper', 'great', 'knew', 'secret', 'secret', 'kept', 'false', 'coin', 'coin', 'coin', 'good', 'coin', 'great', 'knew', 'secret', 'secret', 'kept', 'false']

vocab = Counter(all_words_list)
print(vocab)
# 결과
# Counter({'secret': 6, 'coin': 4, 'barber': 3, 'person': 3, 'good': 3, 'knew': 3, 'kept': 3, 'false': 3, 'paper': 3, 'great': 2, 'huge': 1, 'piece': 1, 'white': 1})

vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도가 많은 5개 단어만 할당
print(vocab)
# 결과
# [('secret', 6), ('coin', 4), ('barber', 3), ('person', 3), ('good', 3)]

word_to_index = {}
i = 0
for word, index in vocab:
    if index > 0:
        i += 1
        word_to_index[word] = i
print(word_to_index)
# 결과
# {'secret': 1, 'coin': 2, 'barber': 3, 'person': 4, 'good': 5}


# NLTK의 FreqDist 사용
from nltk import FreqDist
import numpy as np

vocab = FreqDist(np.hstack(preprocessed_sentence)) # np.hstack : 2차원 배열을 1차원 배열로 변환
print(vocab)
print(vocab["barber"])
# 결과
# <FreqDist with 13 samples and 36 outcomes>
# 3

vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도가 많은 5개 단어만 할당
print(vocab)
# 결과
# [('secret', 6), ('coin', 4), ('barber', 3), ('person', 3), ('good', 3)]

word_to_index = {}
i = 0
for word, index in vocab:
    if index > 0:
        i += 1
        word_to_index[word] = i
print(word_to_index)
# 결과
# {'secret': 1, 'coin': 2, 'barber': 3, 'person': 4, 'good': 5}


# enumerate 사용
test_input = ['a', 'b', 'c', 'd', 'e']

for index, value in enumerate(test_input):
    print("value: {}, index: {}".format(value, index))

# 결과
# value: a, index: 0
# value: b, index: 1
# value: c, index: 2
# value: d, index: 3
# value: e, index: 4


# 케라스 텍스트 전처리 모듈
from tensorflow.keras.preprocessing.text import Tokenizer

preprocessed_sentence = ['barber', 'person', 'good', 'barber', 'huge', 'person', 'barber', 'kept', 'secret', 'huge', 'secret']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentence)

print(tokenizer.word_index)
# 결과
# {'barber': 1, 'person': 2, 'good': 3, 'huge': 4, 'kept': 5, 'secret': 6}

# word
print(tokenizer.word_counts)lp
# 결과
# OrderedDict([('barber', 3), ('person', 2), ('good', 1), ('huge', 2), ('kept', 1), ('secret', 2)])


print(tokenizer.texts_to_sequences(preprocessed_sentence))
# 결과
# [[1], [2], [5], [1], [3], [2], [1], [6], [4], [3], [4]]

print(tokenizer.texts_to_sequences([['barber', 'person'], ['barber', 'good', 'person']]))
# 결과
# [[1, 2], [1, 3, 2]]