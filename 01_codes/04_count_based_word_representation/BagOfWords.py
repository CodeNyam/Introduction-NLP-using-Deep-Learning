'''
Part 4. 카운트 기반 단어 표현
'''

'''
04-01. 백 오브 워드
단어들 순서는 고려하지 않고, 출현 빈도에만 집중하는 텍스트 데이터 수치화 방법

1. BoW의 과정
1) 단어 집합 생성 : 각 단어에 고유 정수 인덱스 부여
2) 각 인덱스 위치에 단어 토큰의 등장 횟수를 기록하는 벡터 생성

각 단어가 등장한 횟수를 수치화하므로 어떤 단어가 얼마나 등장했는지를 기준으로 세울 수 있다.
이를 통해 문서의 성격을 알아낸다. 또, 이런 특성은 문서 간 유사도를 구하는데 사용한다.

예를 들어 미분, 방정식, 부등식 같은 수학 용어가 자주 나온다면 수학 관련 문서로 분류할 수 있다.

'''

# BoW 예시
from konlpy.tag import Okt

okt = Okt()

def build_bag_of_words(document):
    # 온점 제거 및 형태소 분석
    document = document.replace('.', '')
    tokenized_document = okt.morphs(document)

    word_to_index = {}
    bow = []

    for word in tokenized_document:  
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)  
            # BoW에 전부 기본값 1을 넣는다.
            bow.insert(len(word_to_index) - 1, 1)
        else:
            # 재등장하는 단어의 인덱스
            index = word_to_index.get(word)
            # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
            bow[index] = bow[index] + 1

    return word_to_index, bow


doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
vocab, bow = build_bag_of_words(doc1)
print('vocabulary :', vocab)
print('bag of words vector :', bow)
# 결과
# vocabulary : {'정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9}
# bag of words vector : [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]


# 사이킷 런의 CountVectorizer 사용
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()

# 코퍼스로부터 각 단어의 빈도수를 기록
print('bag of words vector :', vector.fit_transform(corpus).toarray()) 

# 각 단어의 인덱스가 어떻게 부여되었는지를 출력
print('vocabulary :',vector.vocabulary_)

# 결과
# bag of words vector : [[1 1 2 1 2 1]]
# vocabulary : {'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}

# CountVectorizer은 띄어쓰기 수준에서 단어를 자르기 때문에 토큰화 품질이 낮다.
# 따라서 한국어에 적용하면 BoW가 잘 구성되지 않을 것이다.
# 앞선 예시 문장에서 "물가상승률"과 "물가상승률은" 같은 단어와 다른 조사로 이루어진다.
# 따라서 서로 다른 단어로 인식하며, 카운트도 각자 된다.

print("-"*100)
# 이를 위해 불용어를 이용해 BoW를 만들어보자
# 1. 사용자 정의 불용어 사용
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])
print('bag of words vector :',vect.fit_transform(text).toarray())
print('vocabulary :',vect.vocabulary_)
# 결과
# bag of words vector : [[1 1 1 1 1]]
# vocabulary : {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}

# 2. CountVectorizer에서 제공하는 불용어 사용
text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
print('bag of words vector :',vect.fit_transform(text).toarray())
print('vocabulary :',vect.vocabulary_)
# 결과
# bag of words vector : [[1 1 1]]
# vocabulary : {'family': 0, 'important': 1, 'thing': 2}

# 3. NLTK 지원 불용어 사용
text = ["Family is not an important thing. It's everything."]
stop_words = stopwords.words("english")
vect = CountVectorizer(stop_words=stop_words)
print('bag of words vector :',vect.fit_transform(text).toarray()) 
print('vocabulary :',vect.vocabulary_)
# 결과
# bag of words vector : [[1 1 1 1]]
# vocabulary : {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}