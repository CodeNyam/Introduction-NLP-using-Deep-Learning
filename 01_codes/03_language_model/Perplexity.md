# Part 3. 언어 모델
# 03-5. 퍼플렉시티(Perplexity, PPL)

## 1. 언어 모델 평가 방법 - PPL
perplexed(헷갈리는)의 의미와 유사하다.

PPL은 문장 길이로 정규화된 문장 확률의 역수이다.

문장 W의 길이가 N일때 PPL은 아래처럼 구한다.
![image](<../../02_images/PPL.png>)


## 2. 분기 계수(Branching factor)
PPL이 선택할 수 있는 경우의 수를 의미한다.
즉, 언어 모델 실행시 PPL이 몇 가지 선택지를 갖는지를 뜻한다.

ex. 언어 모델에서 테스트 데이터 10개를 주고 실험해본 결과, PPL이 10
</br>
다음 단어를 예측하는 모든 시점에서 평균적으로 10개의 단어에서 정답을 고민하고 있다고 볼 수 있음.

따라서 두 언어 모델에서 PPL 계산 후 두 PPL을 비교하여 더 낮은 지표를 갖는 모델이 성능이 좋다고 할 수 있다.

![image](<../../02_images/brancing_factor.png>)

단, PPL이 낮은 것은 테스트 데이터에서 모델이 정확도가 높다는 것이지, 실제 성능이 좋음을 의미하진 않는다.
</br>
또, PPL은 테스트 데이터에 의존하므로 서로 다른 언어 모델을 비교할 때 정량적으로 양이 많고 도메인에 알맞은 테스트 데이터를 선별해야 한다.


## 3. 기존 언어 모델 Vs. 인공 신경망을 이용한 언어 모델
페이스북 AI 연구팀에서 n-gram 모델과 딥러닝 언어 모델의 성능 차이를 표로 공개한 바 있다.

![LLM Research](<https://engineering.fb.com/2016/10/25/ml-applications/building-an-efficient-neural-language-model-over-a-billion-words/>)
- Table 1 참고

일반화된 n-gram 언어 모델의 PPL은 67.6으로 측정되었다.
이외에 인공 신경망 기반 언어 모델들이 n-gram 언어 모델보다 PPL이 낮게 나왔음을 보여준다.


