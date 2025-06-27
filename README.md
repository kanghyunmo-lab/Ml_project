# Quantum Leaf - 암호화폐 자동 트레이딩 시스템

## 프로젝트 개요
Quantum Leaf는 머신러닝 기반의 자동화된 암호화폐 트레이딩 시스템입니다. 시계열 데이터 분석과 강화학습을 결합하여 시장의 추세를 예측하고, 체계적인 리스크 관리 하에서 자동으로 거래를 실행합니다.

## 주요 기능
- **데이터 파이프라인**: 실시간/과거 데이터 수집, 전처리, 피처 엔지니어링 자동화
- **머신러닝 모델**: LightGBM 기반의 다중 분류 모델로 시장 방향성 예측
- **리스크 관리**: ATR 기반 포지션 사이징, 손절매/익절 자동화
- **백테스팅**: 역사적 데이터를 활용한 전략 검증
- **자동 거래**: 바이낸스 선물 거래소와 연동한 자동 매매

## 시작하기

### 사전 요구사항
- Python 3.8+
- Git

### 설치
1. 저장소 클론:
   ```bash
   git clone https://github.com/yourusername/quantum-leaf.git
   cd quantum-leaf
   ```

2. 가상환경 생성 및 활성화:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

4. 환경 변수 설정:
   `.env` 파일을 생성하고 다음 변수들을 설정하세요:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_SECRET_KEY=your_secret_key_here
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

## 사용법

### 데이터 수집
```bash
python -m src.collection.binance_collector
```

### 모델 학습
```bash
python -m src.modeling.trainer
```

### 백테스팅 실행
```bash
python -m src.backtesting.engine
```

### 자동 트레이딩 시작
```bash
python -m src.main
```

## 프로젝트 구조
```
quantum-leaf/
├── data/                   # 데이터 저장소
│   ├── raw/                # 원시 데이터
│   ├── processed/          # 전처리된 데이터
│   └── backtest/           # 백테스팅 결과
├── docs/                   # 문서
├── notebooks/              # Jupyter 노트북
├── src/                    # 소스 코드
│   ├── backtesting/        # 백테스팅 엔진
│   ├── collection/         # 데이터 수집
│   ├── execution/          # 주문 실행
│   ├── modeling/           # 머신러닝 모델
│   ├── monitoring/         # 모니터링 도구
│   ├── processing/         # 데이터 처리
│   ├── utils/              # 유틸리티 함수
│   └── main.py             # 메인 애플리케이션
├── .env.example           # 환경 변수 예시
├── .gitignore
├── README.md
└── requirements.txt
```

## 기여 방법
1. 이슈를 생성하여 작업 내용을 알려주세요.
2. `feature/기능-이름` 브랜치를 생성하여 작업하세요.
3. PR(Pull Request)을 통해 변경사항을 제출해주세요.

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
