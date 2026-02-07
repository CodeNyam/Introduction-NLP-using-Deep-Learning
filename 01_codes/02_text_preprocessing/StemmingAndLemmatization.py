'''
Part 2. 텍스트 전처리
'''

'''
02-03. 어간 추출 and 표제어 추출
1. 표제어 추출
표제어(Lemma) : 기본 사전형 단어
ex. be는 is, are, am의 뿌리 단어이자 표제어

'''
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words_list = ["running", "ate", "cars", "better", "runningly", "fairly", "has", "starting", "dies", "fly"]

print(f"표제어 추출 전 : {words_list}")
print("표제어 추출 후 :", [lemmatizer.lemmatize(word) for word in words_list])
# 결과
# 표제어 추출 전 : ['running', 'ate', 'cars', 'better', 'runningly', 'fairly', 'has', 'starting', 'dies', 'fly']
# 표제어 추출 후 : ['running', 'ate', 'car', 'better', 'runningly', 'fairly', 'ha', 'starting', 'dy', 'fly']
# 의미없는 ha, dy가 추출됨. 이는 lemmatizer가 본래 단어의 품사 정보를 알아야 정확하게 추출하기 때문

print(lemmatizer.lemmatize("dies", 'v'))
# 어간 추출한 결과는 품사 정보가 보존되지 않는다. 또, 추출한 결과가 사전에 존재하지 않는 단어일 때가 많다.


