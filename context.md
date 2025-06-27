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


### 5. 다음에 수행할 작업 (Next Task)

### Phase 1: 기반 공사 (Foundation & Data Pipeline)
**목표:** 프로젝트의 기본 구조를 만들고, 모델 훈련에 사용할 깨끗하고 신뢰성 있는 데이터를 완벽하게 준비한다.

- [ ] **Task 1.1: 프로젝트 초기화 및 문서화**
    - [ ] 로컬에 프로젝트_퀀텀_리프 폴더 생성
    - [ ] Git 저장소 초기화 (git init)
    - [ ] PRD.md 및 docs/ 폴더 내 모든 세부 계획서 문서화 완료
- [ ] **Task 1.2: 데이터 수집 모듈 구현**
    - [ ] src/collection/binance_collector.py 파일 생성
    - [ ] 바이낸스 API를 통해 지정된 기간의 4H OHLCV 데이터를 수집하는 함수 구현
    - [ ] 수집된 데이터를 data/raw/ 폴더에 Parquet 형식으로 저장하는 기능 구현
- [ ] **Task 1.3: 데이터 품질 감사 모듈 구현**
    - [ ] src/processing/audit.py 파일 생성
    - [ ] 원본 데이터의 결측치, 중복, 이상치를 탐지하여 리포트를 생성하는 함수 구현
- [ ] **Task 1.4: 데이터 정제 모듈 구현**
    - [ ] src/processing/clean.py 파일 생성
    - [ ] 감사 리포트를 바탕으로 데이터를 정제하고, data/processed/ 폴더에 저장하는 함수 구현
