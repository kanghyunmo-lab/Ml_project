# 데이터 전략 계획서 (v1.0)

**기준 문서:** PRD v1.1, 최종 실행 계획서

---

## 1. 개요
본 문서는 '프로젝트 퀀텀 리프'의 데이터 파이프라인 구축에 대한 구체적인 전략과 기술적 명세를 정의합니다. 목표는 머신러닝 모델 훈련에 필요한 고품질의 시계열 데이터를 안정적으로 수집, 정제, 저장하는 것입니다.

## 2. 데이터 소스 및 명세
- **데이터 소스:** 바이낸스(Binance) 거래소 API
- **수집 라이브러리:** ccxt
- **대상 자산:** BTC/USDT (선물)
- **시간 단위(Timeframe):** 4시간 봉 (4H)
- **수집 기간:** 가장 최근 7년
- **훈련 데이터 기간:** 수집된 7년 중, 가장 최근 5년의 데이터를 모델 훈련 및 검증에 사용

## 3. 데이터 파이프라인 아키텍처
데이터는 수집 → 품질 감사 → 정제 → 저장의 4단계 파이프라인을 거칩니다.

### 3-1. 데이터 수집 (src/collection/)
- **담당 모듈:** binance_collector.py
- **핵심 기능:** ccxt 라이브러리를 사용하여 바이낸스 API로부터 지정된 기간과 시간 단위의 OHLCV 데이터를 수집
- **출력:** 수집된 원본 데이터는 `data/raw/` 폴더에 Parquet 형식(`btc_usdt_4h_raw.parquet`)으로 저장

### 3-2. 데이터 품질 감사 (src/processing/audit.py)
- **목표:** 원본 데이터의 무결성을 검증하고, 잠재적인 문제점을 사전에 식별
- **담당 모듈:** audit.py의 `run_data_audit()` 함수
- **감사 항목:**
    1. 타임스탬프 연속성 검사: 4시간 간격이 누락된 구간이 있는지 확인
    2. 결측치(NaN) 확인: 각 컬럼(O, H, L, C, V)의 결측치 개수 집계
    3. 중복 데이터 확인: 완전히 동일한 행이 중복으로 존재하는지 확인
    4. 이상치(Outlier) 탐지: 수익률의 표준편차를 기반으로 통계적으로 비정상적인 가격 변동 또는 거래량 급증 탐지
- **출력:** 감사 결과를 담은 텍스트 리포트

### 3-3. 데이터 정제 (src/processing/clean.py)
- **목표:** 품질 감사 결과를 바탕으로, 모델 훈련에 적합한 깨끗한 데이터를 생성
- **담당 모듈:** clean.py의 `clean_ohlcv_data()` 함수
- **처리 프로토콜:**
    1. **중복 데이터:** `pandas.drop_duplicates()`를 사용하여 제거
    2. **결측치:**
        - 가격 (O, H, L, C): 작은 공백(최대 3개 연속)은 **전방 채우기(ffill)**로 처리
        - 거래량 (V): 0으로 채움
        - 위 처리 후에도 남은 결측치가 있는 행은 제거
    3. **이상치:** 탐지된 이상치는 로그에 기록하고, 해당 행을 제거하거나 특수 처리를 고려(초기에는 제거 원칙)
- **출력:** 정제 완료된 데이터는 `data/processed/` 폴더에 Parquet 형식(`btc_usdt_4h_processed.parquet`)으로 저장
