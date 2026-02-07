'''
Part 2. 텍스트 전처리
'''

'''
02-01. 토큰화

단어 토큰화
토큰 기준을 단어로 하는 경우를 말한다.
다만 여기서 단어는 word 외에 단어구, 어절, 형태소 등등이 될 수 있다.
예) "나는 학교에 간다." -> ["나", "는", "학교", "에", "간다", "."]

보통 토큰화 작업은 단순한 공백, 구두점 제거 등으로 끝나지 않는다.
띄어쓰기 단위로 구분되는 영어와 달리 한국어는 띄어쓰기만으로 구분하기 어렵다.
'''

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# 영어에서 단어 토큰화
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
# 아래 의존성은 TensorFlow 2.16 이상 버전 또는 최신 Keras 3에서 deprecated 되어 다른 코드로 대체
# from tensorflow.keras.preprocessing.text import text_to_word_sequence

print('단어 토큰화1 :', word_tokenize("I can't go to school today."))
print('단어 토큰화2 :', WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a place that's not a place."))
# 결과
# 단어 토큰화1 : ['I', 'ca', "n't", 'go', 'to', 'school', 'today', '.']
# 단어 토큰화2 : ['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'place', 'that', "'", 's', 'not', 'a', 'place', '.']

'''
토큰화 작업 주의사항
1. 구두점, 특수 문자를 단순하게 제외하면 안됨
구두점 조차 하나의 토큰으로 분리하기도 함.
예를 들며, 마침표는 문장의 마침을 나타내므로 문장 간 경계를 알 수 있음. 
또, Ph.D, m.p.h 처럼 단어 축약에 사용되거나 01/03/12 같이 날짜 표기에 슬래시가 사용됨

표준 토큰화 예제인 Penn Treebank Tokenizer 규칙을 한 번 살펴보자.
규칙1) 하이픈으로 구성된 단어는 하나로 유지
규칙2) doesn't 같이 어파스트로피가 있는 단어는 분리

'''
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
text = "The U.S.A. isn't the only country in the world. She doen't know that."
print("Treebank 토큰화: ", tokenizer.tokenize(text))
# 결과
# Treebank 토큰화:  ['The', 'U.S.A.', 'is', "n't", 'the', 'only', 'country', 'in', 'the', 'world.', 'She', 'doe', "n't", 'know', 'that', '.']


'''
문장 토큰화
보통 갖고 있는 코퍼스가 정제되지 않은 상태라면, 문장 단위로 토큰화가 필요할 수 있다.
물음표, 느낌표, 마침표 등은 문장 구분자 역할을 해주긴 하나 반드시 그렇진 않다.
ex. IP 192.xxx.xx.xx 서버에 들어가서 로그 파일 좀 저장해줘. 이후에 xxx@gmail.com으로 보내줄래? 내가 점심 사줄게.
'''
from nltk.tokenize import sent_tokenize
text1 = "Dr. Smith graduated from the Univ. of Calif. in 1995. He is now a renowned scientist. He lives in the U.S.A."
text2 = "I love programming! Do you love it too? Let's code together."
print("문장 토큰화 1: ", sent_tokenize(text1))
print("문장 토큰화 2: ", sent_tokenize(text2))
# 결과
# 문장 토큰화 1:  ['Dr. Smith graduated from the Univ.', 'of Calif. in 1995.', 'He is now a renowned scientist.', 'He lives in the U.S.A.']
# 문장 토큰화 2:  ['I love programming!', 'Do you love it too?', "Let's code together."]

# 한국어 토크나이저
# uv pip install kss
import kss

korean_text = "안녕하세요, 저는 딥 러닝에 관심있는 사람입니다. 영어보다 한국어 토큰화가 어렵다네요? 한 번 테스트 해보겠습니다."
print("한국어 문장 토큰화: ", kss.split_sentences(korean_text))
# 결과
# ['안녕하세요, 저는 딥 러닝에 관심있는 사람입니다.', '영어보다 한국어 토큰화가 어렵다네요?', '한 번 테스트 해보겠습니다.']

# 만약에 안쓰면?
print("한국어 토큰화(kss 미사용): ", sent_tokenize(korean_text))
# 결과
# 한국어 토큰화(kss 미사용):  ['안녕하세요, 저는 딥 러닝에 관심있는 사람입니다.', '영어보다 한국어 토큰화가 어렵다네요?', '한 번 테스트 해보겠습니다.']
# 뭐야 똑같자나

# 어절 토큰화는 NLP에서 지양한다. 
# 단어 토큰화와 같지 않은데 이는 한국어가 영어와 달리 교착어이기 때문이다.
# 교착어란 조사, 어미 등을 붙여 말을 만드는 언어를 일컫는다.

'''
교착어의 특성
한국어에는 조사가 많다. 또, 조사 뒤에 띄어쓰기가 바로 붙는다.
NLP에서 서로 다른 조사가 붙어서 다른 단어로 인식되면 자연어를 처리하기 번거롭다.
한국어는 어절이 독립적 단어로 구성되지 않고, 조사 등 무언가에 붙어서 만들어지기에 이를 전부 분리해줘야 한다.

한국어 토큰화에서는 형태소(morpheme)를 반드시 이해해야 한다.
1. 자립 형태소 : 접사, 어미, 조사와 관계없이 자립해서 사용할 수 있는 형태소. 그 자체로 단어가 됨. 
    ex. 체언(명사, 대명사, 수사), 수식 언어(관형사, 부사), 감탄사
2. 의존 형태소 : 다른 형태소와 결합해 사용되는 형태소.
    ex. 접사(접두사, 접미사), 어미, 조사, 어간

Ex. 형태소 단위 분해
"나는 학교에 간다."
- 자립 형태소 : ["나", "학교"]
- 의존 형태소 : ["는", "에", "ㄴ다", "가-", "-다"]

한국어는 띄어쓰기가 영어보다 지켜지지 않는다.
대게 한국어 코퍼스에서는 띄어쓰기가 잘 지켜지지 않는다.
한국어는 모아쓰기 특성이 있기에 띄어쓰기 없이 써도 사람들이 이해할 수 있다.
'''

'''
품사 태깅(Part-of-Speech Tagging)
각 단어가 어떤 품사로 쓰였는지 구분하는 작업
ex. "못" -> 명사 혹은 동작 동사와 사용 가능

'''
import nltk
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize
from nltk import pos_tag # NLTK에서는 Penn Treebank 기준 품사 태깅 사용

text = "I can't go to school today."
tokenized_text = word_tokenize(text)

print("단어 토큰화 : ", tokenized_text)
print("품사 태깅 : ", pos_tag(tokenized_text))

# 결과
# 단어 토큰화 :  ['I', 'ca', "n't", 'go', 'to', 'school', 'today', '.']
# 품사 태깅 :  [('I', 'PRP'), ('ca', 'MD'), ("n't", 'RB'), ('go', 'VB'), ('to', 'TO'), ('school', 'NN'), ('today', 'NN'), ('.', '.')]

from konlpy.tag import Okt # Open Korean Text
from konlpy.tag import Kkma # 꼬꼬마
# 이외에 Mecab, Hannanum, Komoran
from konlpy.tag import Mecab

okt = Okt()
kkma = Kkma()

print("Okt 형태소 분석: ", okt.morphs("나는 연휴에 여행갈테야 갈테야 여행을 갈테야."))
print("Okt 품사 태깅: ", okt.pos("나는 연휴에 여행갈테야 갈테야 여행을 갈테야."))
print("Okt 명사 추출 : ", okt.nouns("나는 연휴에 여행갈테야 갈테야 여행을 갈테야."))
# 결과
# Okt 형태소 분석:  ['나', '는', '연휴', '에', '여행', '갈테야', '갈테야', '여행', '을', '갈테야', '.']
# Okt 품사 태깅:  [('나', 'Noun'), ('는', 'Josa'), ('연휴', 'Noun'), ('에', 'Josa'), ('여행', 'Noun'), ('갈테야', 'Verb'), ('갈테야', 'Verb'), ('여행', 'Noun'), ('을', 'Josa'), ('갈테야', 'Verb'), ('.', 'Punctuation')]
# Okt 명사 추출 :  ['나', '연휴', '여행', '여행']

print("Kkma 형태소 분석: ", kkma.morphs("나는 연휴에 여행갈테야 갈테야 여행을 갈테야."))
print("Kkma 품사 태깅: ", kkma.pos("나는 연휴에 여행갈테야 갈테야 여행을 갈테야."))
print("Kkma 명사 추출 : ", kkma.nouns("나는 연휴에 여행갈테야 갈테야 여행을 갈테야."))
# 결과
# Kkma 형태소 분석:  ['나', '는', '연휴', '에', '여행', '갈', 'ㄹ', '터', '이', '야', '가', 'ㄹ', '터', '이', '야', '여행', '을', '갈', 'ㄹ', '터', '이', '야', '.']
# Kkma 품사 태깅:  [('나', 'NP'), ('는', 'JX'), ('연휴', 'NNG'), ('에', 'JKM'), ('여행', 'NNG'), ('갈', 'VV'), ('ㄹ', 'ETD'), ('터', 'NNB'), ('이', 'VCP'), ('야', 'EFN'), ('가', 'VV'), ('ㄹ', 'ETD'), ('터', 'NNB'), ('이', 'VCP'), ('야', 'EFN'), ('여행', 'NNG'), ('을', 'JKO'), ('갈', 'VV'), ('ㄹ', 'ETD'), ('터', 'NNB'), ('이', 'VCP'), ('야', 'EFN'), ('.', 'SF')]
# Kkma 명사 추출 :  ['나', '연휴', '여행', '터']

# 추가 분석
mecab = Mecab()
# print("Mecab 형태소 분석: ", mecab.morphs("나는 연휴에 여행갈테야 갈테야 여행을 갈테야."))
# print("Mecab 품사 태깅: ", mecab.pos("나는 연휴에 여행갈테야 갈테야 여행을 갈테야."))
# print("Mecab 명사 추출 : ", mecab.nouns("나는 연휴에 여행갈테야 갈테야 여행을 갈테야."))
# 결과