'''
Part1. 기본 과정
'''

'''
자연어 처리
- 음성 인식, 요약, 챗봇, 감성 분성, 텍스트 분류 작업에 사용
'''

'''
필요 프레임 워크
Notes.
1. 모드 uv 패키지 매니저를 사용해 빠르게 설치함
2. 책에서는 pytorch가 아닌 tensorflow 사용.(다만 대세는 pytorch)

1. 텐서 플로우
- uv pip install tensorflow
- 2015년 구글에서 발표한 머신 러닝 오픈소스

2. 케라스
- uv pip install keras
- tensorflow에 대해 추상화된 API 제공
- 순수 케라스도 있지만, 주로 tensorflow에서 제공하는 API 사용
- 따라서 tf.keras로 호출 후 사용함

3. 젠심
- uv pip install gensim
- 머신러닝을 이용해 토픽 모델링 및 자연어 처리를 수행하는 오픈소스
- 책에서 젠심으로 Word2Vec 등 모델을 구현할 예정

4. 사이킷런
- uv pip install scikit-learn
- 나이브 베이즈 분류, 서포트 벡터 머신 등 다양한 머신 러닝 모듈을 불러올 수 있음.

5. 주피터 노트북
- uv pip install jupyterlab

6. nltk
- uv pip install nltk

7. konlpy
- uv pip install konlpy

8. pandas
- uv pip install pandas

9. numpy
- uv pip install numpy

10. matplotlib
- uv pip install matplotlib 

'''

'''
머신러닝 워크 플로우
1. 수집
머신러닝에 이용할 학습 데이터 수집
자연어 처리에서는 말뭉치 또는 코퍼스(corpus)라고 부름
코퍼스 : 조사나 연구 목적으로 수집한 특정 도메인 텍스트의 집합
형식은 txt, csv, xml 등 다양하며, 출처도 영화 리뷰, 음성 등 다양함

2. 점검 및 탐색
수집한 데이터를 점검하고 파악하는 단계
데이터 구조, 노이즈 분석, 데이터 정재 방법에 대해 논의
이 단계에서 탐색적 데이터 분석(EDA, Exploratory Data Analysis) 수행
EDA는 독립 변수, 종속 변수, 변수 유형, 데이터 유형 등등을 파악하고 데이터에 내재하는 구조적 관계를 알아내고자 함

3. 전처리 및 정재
수집한 데이터를 머신러닝에 적합한 형태로 변환하는 단계이며, 가장 많은 과정이 포함됨
자연어 처리에선 토큰화, 정제, 정규화, 불용어 제거 등의 단계 포함

4. 모델링 및 훈련
머신러닝 알고리즘을 적용해 모델을 생성하고 훈련하는 단계
이때 모든 데이터를 기계에게 training 시키면 안된다. 
Train/Validation/Test 3가지로 나누어야함

5. 평가

6. 배포
성공적으로 훈련이 되었다고 판단시 완성된 모델을 시중에 배포
다만, 피드백이 주어져 업데이트 해야한다면, 다시 1번 단계로 돌아감
'''