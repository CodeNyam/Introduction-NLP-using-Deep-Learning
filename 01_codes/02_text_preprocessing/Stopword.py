'''
Part 2. 텍스트 전처리
'''

'''
02-04. 불용어 
조사, 접미사 같이 자주 등장하지만, 도움되지 않는 단어들
'''

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt

# 영어 불용어 확인
stop_words_list = stopwords.words('english')
print("영어 불용어 갯수:", len(stop_words_list))
print("영어 불용어 20개 출력: ", stop_words_list[:20])
# 결과
# 영어 불용어 갯수: 198
# 영어 불용어 20개 출력:  ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been']


# 문장에서 불용어 제거하기
example_sentence = "This is a sample sentence, showing off the stop words filtration."
stop_words = set(stopwords.words('english'))
example_word_tokens = word_tokenize(example_sentence)

filtered_sentence = []
for word in example_word_tokens:
    if word not in stop_words:
        filtered_sentence.append(word)
print("원래 문장:", example_word_tokens)
print("불용어 제거 후:", filtered_sentence)
# 결과
# 원래 문장: ['This', 'is', 'a', 'sample', 'sentence', ',', 'showing', 'off', 'the', 'stop', 'words', 'filtration', '.']
# 불용어 제거 후: ['This', 'sample', 'sentence', ',', 'showing', 'stop', 'words', 'filtration', '.']

# 한국어 불용어 사이트
# https://www.ranks.nl/stopwords/korean