# Part 3. 언어 모델

## 03-2. 통계적 언어 모델(SLM)
### 조건부 확률을 이용한 문장에서 확률
- 조건부 확률 공식
![image](<02_images/conditional_probability.png>)
</br>
- 조건에 대한 확률
![image](</02_images/word_sequence_probability_2.png>)

### 카운터 기반 접근
SLM은 이전 단어에서 다음 단어에 대한 확률을 카운트 기반으로 구한다.
![image](</Users/mac/Desktop/Introduction-NLP-using-Deep-Learning/02_images/pro_based_count.png>)

### 카운터 기반 접근법의 한계 - 희소 문제(Sparsity Problem)
기계에게 많은 코퍼스를 훈련시켜 언어 모델의 확률을 확률 분포에 맞추는 것이 목표이다.
카운트 기반으로 접근하면, 학습에 필요한 코퍼스가 매우 많다.
위 예시에서 코퍼스 내에 "An adorable little boy is" 단어 시퀀스가 없다면, 확률은 0이다.
코퍼스에 없는 단어라고 해서 확률을 0으로 정의하는 것은 옳지 않다. 현실에 존재하는 단어이며 문법적으로 올바른 구조이기 때문이다.
이처럼 충분한 데이터를 학습하지 못해 언어 모델이 정확히 모델링하지 못하는 것을 희소 문제라고 한다.
</br>
위 문제를 완화하기 위해 n-gram 언어 모델이나 스쿠딩, 백오프 같은 일반화(generalization)기법이 존재한다.
</br>
다만 실질적 해법이 되지 못했으며, 이러한 한계로 인해 언어 모델은 SLM에서 인공 신경망 언어 모델로 트렌드가 바뀐다.