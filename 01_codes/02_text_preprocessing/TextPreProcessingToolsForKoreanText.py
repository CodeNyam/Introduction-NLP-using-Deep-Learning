'''
Part 2. 텍스트 전처리
'''

'''
02-10. 한국어 텍스트 전처리 도구
'''

# PyKoSpacing
# pip install git+https://github.com/haven-jeon/PyKoSpacing.git

# Py-hanspell
# 2024/03/03 기준 잘 동작하지 않는다는데, 예제는 많이 나옴
# 보니까 업데이트가 3년 전이 마지막임.. 
# 파이썬 2.7, 3.4에서 호환된다고 하니 최근 버전에서는 안될 가능성 높음
# pip install git+https://github.com/ssut/py-hanspell.git

# SOYNLP
# pip install soynlp
# 비지도 학습으로 단어 토큰화
# 자주 등장하는 단어를 단어로 분석
# 내부적으로 단어 점수 표로 동작 - 응집 확률(cohesion probability) & 브랜칭 엔트로피(branching entropy) 활용
# 신조어 학습에 유용
# 반복 문자 정제에도 사용
from soynlp.normalizer import *

print(emoticon_normalize('앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠㅠㅠ', num_repeats=2))
# 아ㅋㅋ영화존잼쓰ㅠㅠ
# 아ㅋㅋ영화존잼쓰ㅠㅠ
# 아ㅋㅋ영화존잼쓰ㅠㅠ
# 아ㅋㅋ영화존잼쓰ㅠㅠ