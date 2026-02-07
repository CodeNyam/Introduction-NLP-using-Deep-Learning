# Introduction-NLP-using-Deep-Learning

위키독스 <딥러닝을 이용한 자연어 처리 입문> 내용 정리 및 기록

## 입문자를 위한 조언 Q&A

- text-to-sql 모델 파인튜닝
- RAFT 논문 기반 LLM 파인튜닝
- 초기에 데이터셋이 없을 경우 허깅페이스나 AI Hub에서 초기 시드 데이터 서치 -> 범용 AI 모델로 정제 -> LLM 파인튜닝 경험
  - 비용이 고민될 땐 Runpod 같은 클라우드 서비스 이용
- NLP 엔지니어로서... 트랜스포머, T5, LLaMa, Deepspeed, LoRA tuning, FlashAttention, RAG, Agent 고도화를 학습해보기
  - 위 방법에는 청킹 전략, GraphRAG, ReACT agent, Function Calling 등이 사용

# 아나콘다 설치(macOS)

1. brew로 아나콘다 설치
   ''' code
   brew install --cask anaconda
   '''

2. 환경변수 설정
   ''' code
   echo 'export PATH="/opt/homebrew/anaconda3/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   '''

3. 아나콘다 터미널 초기화
   ''' code
   conda init zsh
   '''

4. 아나콘다 버전 확인
   ''' code
   conda --version
   '''

# 가상환경 설정

1. 원하는 이름으로 가상환경 설정

```code
conda create -n introduce_nlp python=3.13.2
```

2. 디렉토리 소유권 설정
   상황) 디렉토리 소유권이 내가 쓰는 유저가 아니였음. .conda 디렉터리 소유권을 현재 유저로 변경함

```code
   sudo chown -R $(whoami) ~/.conda
```

- 이후 약관이 나타나면 a 입력 후 필요 패키지 설치

## 가상환경 활성화

```code
   conda activate introduce_nlp
```

## 가상환경 비활성화

```code
   conda deactivate
```
