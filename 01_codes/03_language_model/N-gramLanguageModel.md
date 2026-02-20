# Part 3. 언어 모델

## 03-03. N-gram 언어 모델
n-gram 언어 모델은 카운트에 기반한 SLM이다.
</br>
다만 앞서 배운 모델과 달리 이전에 등장한 모든 단어가 아닌 일부만 고려한다.
</br>
이때 몇 개의 일부 단어를 보느냐를 결정하는데 이 변수가 n이다.

## 1. 코퍼스에서 카운트 못하는 단어 수 감소시키기
SLM의 한계는 훈련 코퍼스에 문장이 없다면, 해당 문장에 대한 확률을 구할 수 없다는 점이다.
</br>
또, 문장이 길어질수록 코퍼스에서 그 문장이 존재하지 않을 가능성이 높다.
즉, 카운트할 수 없다.
</br>
이때 다음처럼 참고하는 단어를 줄이면 카운트 할 가능성을 높일 수 있다.
</br>
"An adorable little boy"가 나왔을때 "is"가 나올 확률을 "boy"가 나왔을때 확률과 동일시하면 어떨까?
<br>
"An ~" 문장보다 "boy is"가 코퍼스에 있을 확률이 더 높다.

![image](<../../02_images/reduce_not_countable_words.png>)


## 2. N-gram
앞서 보았듯이, 긴 단어 조합에서 임의의 단어 일부만 기준점으로 사용하는 것을 n-gram이라고 한다.
</br>
코퍼스 내에서 n개의 단어 뭉치를 하나의 토큰으로 간주한다.
- ex. "An adorable little boy is spreading smiles"의 n-gram을 모두 구해보면...
1. unigrams : an, adorable, little, boy, is, spreading, smiles
2. bigrams : an adorable, adorable little, little boy, boy is, is spreading, spreading smiles
3. trigrams : an adorable little, adorable little boy, little boy is, boy is spreading, is spreading smiles
4. 4-grams : an adorable little boy, adorable little boy is, little boy is spreading, boy is spreading smiles

- n=1 => 유니그램
- n=2 => 바이그램
- n=3 => 트라이그램
- n=4 => 그냥 앞에 숫자랑 gram만 붙임, 종종 1-gram, 2-gram, 3-gram으로도 부름
</br>
N-gram을 통해 다음 단어를 예측시 빈칸 앞의 (n-1)개의 단어만 고려한다.
ex. "An adorable little boy is spreading"에서 4-gram 적용시

![image](<../../02_images/n_gram_example.png>)

(insult는 모욕하다인데...)

## N-gram 언어 모델의 한계
앞서 예시에서 의문이 남는다. an adorable little 소년이 과연 모욕을 했을까?
<br>
insults보다 smiles가 의미적으로 맞지 않나?
</br>
코퍼스 데이터를 어떻게 가정하고 구축했느냐에 따라서 확률은 다를 것이다.
다만 인간이 보기에 sematic이 맞지 않는다고 느낄 수 있다.

이처럼 N-gram 방식은 전체 단어의 context를 파악할 수 없어 의도한대로 문장을 끝맺지 못할 수 있다.

즉, 전체 문장을 고려하는 언어 모델보다 정확도와 성능이 떨어질 수 밖에 없다.

### 1. 희소 문제
1. n을 선택하는 것은 trade-off가 존재
- n을 크게 잡으면, 컨텍스트가 늘어나므로 성능은 좋아지나 모델 크기가 증가함
- n을 작게 잡으면, 모델 크기는 줄어드나 문장의 핵심 내용을 컨텍스트로 입력하지 못할 가능성이 있음

n을 작게 잡으면 코퍼스에서 카운트는 잘 되지만, 현실 언어 세계에서 확률 분포와 동떨어진다.

스탠포드 대학교에서 이에 대한 연구를 진행했다.
</br>
월스트리트 저널에서 3,800만 개 단어 토큰 추출 후 n-gram 언어 모델에 학습시켰다.
</br>
이후 1,500만 개 테스트 데이터로 실험해보고 각 그램에서 퍼플렉시티를 측정했다.
(퍼플렉시티가 낮을 수록 성능이 좋음)

- Unigram : 962
- Bigram : 170
- Trigram : 109

### 참고자료
- 스탠포드의 실험과 함께 N-gram 언어 모델 연구가 담겨있음
- 위키북스에서 언급된 스무딩, 백오프 개념 탑재, 인터폴레이션(보간법) 방식도 제시

[n-gram Language Model Paper by Stanford Universiry](https://web.stanford.edu/~jurafsky/slp3/3.pdf)


## 4. 도메인에 맞는 코퍼스 수집
어떤 도메인인지에 따라서 코퍼스의 확률 분포는 달라진다.
</br>
해당 도메인에서 사용할 코퍼스를 구하거나 구축함에 따라서 언어 모델의 성능이 높아진다.
