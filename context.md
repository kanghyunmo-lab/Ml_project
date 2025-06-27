# 프로젝트 퀀텀 리프: AI 작업 브리핑 (CONTEXT.md)

## [PART 1] 핵심 원칙 (Project Constitution)
*(이 부분은 프로젝트의 '헌법'으로, 거의 수정되지 않습니다)*

### 1. 프로젝트 목표 및 성공 기준 (KPIs)
- **목표**: 4시간봉 기반 BTC/USDT 선물 스윙 트레이딩 자동화 시스템 구축
- **핵심 성공 지표 (KPIs)**: 워크 포워드 분석 결과, 아래 기준을 모두 충족해야 함.
  - Sharpe Ratio > 1.2
  - Max Drawdown < 25%
  - Calmar Ratio > 1.0
  - Profit Factor > 1.75
  - Walk-Forward Efficiency > 60%

### 2. 핵심 아키텍처 및 기술 스택 (Core Architecture & Tech)
- **설계 사상**: 모듈형 이벤트 기반 아키텍처 (Modular, Event-Driven Architecture)
- **기술 스택**: Python, pandas, ccxt, LightGBM, Backtrader, Optuna, APScheduler
- **코딩 규약**: CONVENTIONS.md 파일의 네이밍/설계 원칙을 따름.

### 3. 핵심 리스크 및 거래 전략 (Core Risk & Trade Strategy)
- **포지션 사이징**: 명목 가치 기반 2% 리스크 규칙
- **손절매 (Stop-Loss)**: 2배 ATR 기반
- **수익 실현 (Take-Profit)**: 손절매 거리의 1.5배 (손익비 1:1.5 고정). '삼중 장벽 기법'에 의해 통합 관리됨.
- **레버리지**: 최대 3배로 제한.
- **절대 원칙**: '절대 청산 회피 메커니즘' 필수. 모든 거래 전, 손절 가격과 예상 청산 가격을 비교/검증함.

---

## [PART 2] 현재 상태 보고 (Current Status)
*(이 부분은 Task를 완료할 때마다 업데이트됩니다)*

### 4. 현재까지 완료된 작업
### Phase 1: 기반 공사 (Foundation & Data Pipeline)
**목표:** 프로젝트의 기본 구조를 만들고, 모델 훈련에 사용할 깨끗하고 신뢰성 있는 데이터를 완벽하게 준비한다.

- [X] **Task 1.1: 프로젝트 초기화 및 문서화**
    - [X] 로컬에 프로젝트_퀀텀_리프 폴더 생성
    - [X] Git 저장소 초기화 (git init)
    - [X] PRD.md 및 docs/ 폴더 내 모든 세부 계획서 문서화 완료
- [X] **Task 1.2: 데이터 수집 모듈 구현**
    - [X] src/collection/binance_collector.py 파일 생성
    - [X] 바이낸스 API를 통해 지정된 기간의 4H OHLCV 데이터를 수집하는 함수 구현
    - [X] 수집된 데이터를 data/raw/ 폴더에 Parquet 형식으로 저장하는 기능 구현
- [X] **Task 1.3: 데이터 품질 감사 모듈 구현**
    - [X] src/processing/audit.py 파일 생성
    - [X] 원본 데이터의 결측치, 중복, 이상치를 탐지하여 리포트를 생성하는 함수 구현
- [x] **Task 1.4: 데이터 정제 모듈 구현**
    - [x] src/processing/clean.py 파일 생성
    - [x] 감사 리포트를 바탕으로 데이터를 정제하고, data/processed/ 폴더에 저장하는 함수 구현

### Phase 2: 핵심 엔진 개발 (Core Engine Development)
**목표:** 정제된 데이터를 바탕으로 예측 모델을 만들고, 그 성능을 엄격하게 검증할 수 있는 백테스팅 엔진을 완성한다.

- [X] **Task 2.1: 피처 엔지니어링 모듈 구현**
    - [X] src/processing/feature_engineer.py 파일 생성
    - [X] PRD에 명시된 모든 기술적 지표 및 파생 변수(ATR, RSI, MACD 등)를 계산하여 데이터프레임에 추가하는 함수 구현
    - [X] 예외 처리 및 안정성 강화
    - [X] 테스트 케이스 작성 및 검증 완료

- [X] **Task 2.2: 타겟 레이블링 모듈 구현**
    - [X] src/modeling/labeler.py 파일 생성
    - [X] '삼중 장벽 기법'에 따라 수익실현/손절매/시간제한 라인을 설정하고, 1(매수 성공), -1(손절), 0(횡보)으로 레이블링하는 함수 구현
    - [X] 실제 바이낸스 데이터를 활용한 통합 테스트 완료
    - [X] 시각화 기능을 포함한 예제 스크립트 작성

### 5. 다음에 수행할 작업 (Next Task)

- [X] **Task 2.3: 모델 훈련 파이프라인 구현**
    - [X] src/modeling/trainer.py 파일 생성
    - [X] TimeSeriesSplit을 이용한 시계열 교차검증 구조 구현
    - [X] Optuna를 사용하여 LightGBM 모델의 하이퍼파라미터 최적화
    - [X] 모델 저장 및 로드 기능 구현
    - [X] 단위 테스트 작성 (tests/unit/modeling/test_trainer.py) - 모든 테스트 케이스 통과 확인 완료
    - [X] 예제 스크립트 추가 (examples/model_training_example.py)
    - [X] 테스트 커버리지 개선 및 버그 수정 완료
      - `test_evaluate`: 다중 클래스(-1, 0, 1) 평가 로직 검증
      - `test_save_and_load_model`: 모델 저장/로드 기능 검증
- [ ] **Task 2.4: 백테스팅 엔진 구현**
    - [ ] src/backtesting/engine.py 파일 생성
    - [ ] Backtrader를 사용하여, 훈련된 모델의 예측 신호를 바탕으로 거래를 시뮬레이션하는 엔진 구현
    - [ ] 수수료, 슬리피지, 레버리지, 펀딩비 등 현실적인 마찰 비용을 시뮬레이션에 포함
    - [ ] PRD에 명시된 모든 KPI(샤프 지수, MDD 등)를 계산하여 최종 성과 리포트를 출력하는 기능 구현