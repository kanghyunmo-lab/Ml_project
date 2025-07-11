# Quantum Leaf Project: Final Execution Plan (v1.0)

> **Document Purpose:**
> 본 문서는 'PRD v1.1'에 명시된 모든 요구사항을 충족시키기 위한, 시작부터 끝까지의 모든 개발 단계를 정의하는 종합 실행 계획서입니다. 이 계획은 아키텍처를 중심으로 구성되며, 체크리스트 형식을 통해 진행 상황을 추적 관리할 수 있도록 설계되었습니다.

---

## 1. 최종 시스템 아키텍처 (System Architecture)

본격적인 개발에 앞서, 우리가 구축할 시스템의 전체 구조(아키텍처)를 명확히 정의합니다. 모든 개발은 아래 아키텍처에 따라 각 모듈을 완성하고 조립하는 방식으로 진행됩니다.

### 1-1. 핵심 설계 사상
- **모듈형 이벤트 기반 아키텍처:** 각 기능(데이터, 모델, 주문 등)은 독립적인 모듈로 개발하며, 서로 느슨하게 연결되어 유지보수성과 확장성을 극대화합니다.

### 1-2. 폴더 및 모듈 구조 (실제 구현)
```
/ (root)
├── config/
│   └── data_config.yaml          # 데이터 및 모델 설정 파일
├── data/                          # 데이터 저장 디렉토리
│   ├── raw/                      # 원본 데이터
│   ├── processed/                # 전처리된 데이터
│   └── models/                   # 훈련된 모델 저장
├── src/                          # 소스 코드
│   ├── collection/               # 데이터 수집
│   │   └── binance_collector.py  # 바이낸스 API 데이터 수집
│   │
│   ├── processing/             # 데이터 처리
│   │   ├── __init__.py
│   │   ├── audit.py              # 데이터 품질 감사
│   │   ├── clean.py              # 데이터 정제
│   │   └── feature_engineer.py   # 기술적 지표 기반 피처 생성
│   │
│   ├── modeling/               # 모델링 관련
│   │   ├── __init__.py
│   │   ├── labeler.py            # 삼중 장벽 기법 레이블링
│   │   ├── trainer.py            # LightGBM 모델 훈련/최적화
│   │   └── predictor.py          # 훈련된 모델로 예측
│   │
│   ├── backtesting/            # 백테스팅
│   │   ├── __init__.py
│   │   └── engine.py             # Backtrader 기반 백테스팅 엔진
│   │
│   ├── execution/              # 주문 실행 (구현 예정)
│   │   ├── __init__.py
│   │   ├── risk_manager.py       # 리스크 관리
│   │   └── order_handler.py      # 주문 처리
│   │
│   ├── monitoring/             # 모니터링 (구현 예정)
│   │   ├── __init__.py
│   │   ├── logger.py            # 로깅 설정
│   │   └── alerter.py           # 알림 전송
│   │
│   └── utils/                  # 유틸리티
│       ├── __init__.py
│       └── config_loader.py     # 설정 파일 로더
│
├── scripts/                     # 실행 스크립트
│   ├── run_feature_engineering.py  # 피처 엔지니어링 실행
│   ├── run_labeling.py            # 레이블 생성 실행
│   └── train_model.py             # 모델 훈련 실행
│
├── tests/                       # 테스트 코드
│   ├── unit/                     # 단위 테스트
│   └── integration/              # 통합 테스트
│
├── .env                        # 환경 변수
├── requirements.txt              # 파이썬 의존성
└── README.md                    # 프로젝트 설명서
```

---

## 2. 단계별 상세 작업 계획 (Phased Task Plan)

프로젝트를 4개의 주요 단계(Phase)로 나누어 진행합니다. 각 단계의 모든 체크리스트가 완료되어야 다음 단계로 넘어갈 수 있습니다.

### Phase 1: 기반 공사 (Foundation & Data Pipeline) ✅ 완료
**목표:** 프로젝트의 기본 구조를 만들고, 모델 훈련에 사용할 깨끗하고 신뢰성 있는 데이터를 완벽하게 준비한다.

- [x] **Task 1.1: 프로젝트 초기화 및 문서화**
    - [x] 로컬에 프로젝트_퀀텀_리프 폴더 생성
    - [x] Git 저장소 초기화 (git init)
    - [x] PRD.md 및 docs/ 폴더 내 모든 세부 계획서 문서화 완료
- [x] **Task 1.2: 데이터 수집 모듈 구현**
    - [x] src/collection/binance_collector.py 파일 생성
    - [x] 바이낸스 API를 통해 지정된 기간의 4H OHLCV 데이터를 수집하는 함수 구현
    - [x] 수집된 데이터를 data/raw/ 폴더에 Parquet 형식으로 저장하는 기능 구현
- [x] **Task 1.3: 데이터 품질 감사 모듈 구현**
    - [x] src/processing/audit.py 파일 생성
    - [x] 원본 데이터의 결측치, 중복, 이상치를 탐지하여 리포트를 생성하는 함수 구현
- [x] **Task 1.4: 데이터 정제 모듈 구현**
    - [x] src/processing/clean.py 파일 생성
    - [x] 감사 리포트를 바탕으로 데이터를 정제하고, data/processed/ 폴더에 저장하는 함수 구현

### Phase 2: 핵심 엔진 개발 (Core Engine Development) ✅ 완료
**목표:** 정제된 데이터를 바탕으로 예측 모델을 만들고, 그 성능을 엄격하게 검증할 수 있는 백테스팅 엔진을 완성한다.

- [x] **Task 2.1: 피처 엔지니어링 모듈 구현**
    - [x] src/processing/feature_engineer.py 파일 생성
    - [x] PRD에 명시된 모든 기술적 지표 및 파생 변수(ATR, RSI, MACD 등)를 계산하여 데이터프레임에 추가하는 함수 구현
    - [x] 예외 처리 및 안정성 강화
    - [x] 테스트 케이스 작성 및 검증 완료

- [x] **Task 2.2: 타겟 레이블링 모듈 구현**
    - [x] src/modeling/labeler.py 파일 생성
    - [x] '삼중 장벽 기법'에 따라 수익실현/손절매/시간제한 라인을 설정하고, 1(매수 성공), -1(손절), 0(횡보)으로 레이블링하는 함수 구현
    - [x] 실제 바이낸스 데이터를 활용한 통합 테스트 완료
    - [x] 시각화 기능을 포함한 예제 스크립트 작성

- [x] **Task 2.3: 모델 훈련 파이프라인 구현**
    - [x] src/modeling/trainer.py 파일 생성
    - [x] TimeSeriesSplit을 이용한 시계열 교차검증 구조 구현
    - [x] Optuna를 사용하여 LightGBM 모델의 하이퍼파라미터 최적화
    - [x] 모델 저장 및 로드 기능 구현
    - [x] 단위 테스트 작성 (tests/unit/modeling/test_trainer.py)
    - [x] 예제 스크립트 추가 (examples/model_training_example.py)

- [x] **Task 2.4: 백테스팅 엔진 구현**
    - [x] src/backtesting/engine.py 파일 생성
    - [x] Backtrader를 사용하여, 훈련된 모델의 예측 신호를 바탕으로 거래를 시뮬레이션하는 엔진 구현
    - [x] 수수료, 슬리피지, 레버리지, 펀딩비 등 현실적인 마찰 비용을 시뮬레이션에 포함
    - [x] PRD에 명시된 모든 KPI(샤프 지수, MDD 등)를 계산하여 최종 성과 리포트를 출력하는 기능 구현
    - [x] 단위 테스트 작성 (tests/unit/backtesting/test_engine.py)
    - [x] 예제 스크립트 추가 (examples/backtest_example.py)

### Phase 3: 자동화 시스템 구축 (Automation & Execution)
**목표:** 검증된 모델과 전략을 사람의 개입 없이 24시간 자동으로 실행할 수 있는 완전한 시스템으로 조립한다.

- [ ] **Task 3.1: 리스크 관리 모듈 구현**
    - [ ] src/execution/risk_manager.py 파일 생성
    - [ ] '명목 가치 기반 2% 룰'에 따라 포지션 규모를 계산하는 함수 구현
    - [ ] '절대 청산 회피' 로직을 구현하여, 모든 거래 전에 손절 가격과 예상 청산 가격을 비교/검증하는 기능 구현
- [ ] **Task 3.2: 주문 실행 모듈 구현**
    - [ ] src/execution/order_handler.py 파일 생성
    - [ ] ccxt를 사용하여 바이낸스 선물 시장에 격리 모드 및 레버리지를 설정하는 기능 구현
    - [ ] 리스크 관리 모듈의 검증을 통과한 거래를 실제 지정가/시장가 주문으로 전송하는 기능 구현
    - [ ] OCO 주문을 활용하여 익절/손절 주문을 자동으로 관리하는 기능 구현
- [ ] **Task 3.3: 모니터링 시스템 구현**
    - [ ] src/monitoring/logger.py 파일에 중앙 로깅 설정 구현
    - [ ] src/monitoring/alerter.py 파일에 텔레그램으로 긴급 오류 알림을 보내는 함수 구현
- [ ] **Task 3.4: 메인 실행 스크립트 작성**
    - [ ] main.py 파일 작성
    - [ ] APScheduler를 사용하여 정해진 시간(4시간마다)에 '데이터 수집 → 피처 생성 → 모델 예측 → 리스크 검증 → 주문 실행'의 전체 워크플로우를 실행하는 로직 구현

### Phase 4: 실전 투입 및 운영 (Deployment & Go-Live)
**목표:** 완성된 시스템을 실제 시장에 안전하게 투입하고, 지속적으로 운영 및 관리한다.

- [ ] **Task 4.1: 보안 및 서버 배포**
    - [ ] 모든 API 키와 민감 정보를 .env 파일을 통해 환경 변수로 분리
    - [ ] AWS EC2 등 클라우드 서버에 프로젝트 배포
    - [ ] systemd 서비스를 등록하여 시스템이 비정상 종료 시 자동으로 재시작되도록 설정
- [ ] **Task 4.2: 페이퍼 트레이딩 실행 (기술 검증)**
    - [ ] 바이낸스 테스트넷 환경에서 시스템을 최소 1개월 이상 가동
    - [ ] API 연결, 주문 생애주기, 로깅, 알림 등 모든 기술적 요소가 오류 없이 작동하는지 검증
- [ ] **Task 4.3: 인큐베이션 실행 (현실 마찰 측정)**
    - [ ] 총예산의 1~5% 소액 자본으로 실거래 시작
    - [ ] 실제 발생하는 슬리피지와 수수료를 측정하고, 백테스트 결과와 비교하여 '성과 일치 비율'이 80% 이상인지 확인
- [ ] **Task 4.4: 최종 Go/No-Go 결정**
    - [ ] PRD에 명시된 '최종 Go/No-Go 결정 매트릭스' 체크리스트를 기반으로 모든 항목을 최종 점검
    - [ ] 모든 기준 충족 시, 계획된 전체 자본으로 시스템을 본격 가동
