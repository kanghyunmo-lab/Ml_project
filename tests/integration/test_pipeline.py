"""
End-to-end 테스트 스크립트: 데이터 수집부터 백테스트까지 전체 파이프라인 테스트
"""
import os
import sys
import pytest
import pandas as pd
from pathlib import Path

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 테스트에 필요한 모듈 임포트
try:
    from src.collection.binance_collector import fetch_historical_data
    from src.processing.audit import run_data_audit
    from src.processing.clean import clean_ohlcv_data
    from src.processing.feature_engineer import generate_features
    from src.modeling.predictor import load_model, predict
    from src.backtesting.engine import BacktestEngine
    from src.config import DATA_DIR, MODELS_DIR
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

# 테스트에 필요한 설정
TEST_SYMBOL = "BTC/USDT"
TEST_TIMEFRAME = "4h"
TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2024-01-31"  # 1개월 데이터로 테스트

# 종속성 확인
pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES,
    reason="필요한 의존성이 설치되지 않았습니다."
)

# 테스트 데이터 경로
TEST_DATA_DIR = DATA_DIR / "test"
TEST_RAW_DATA_PATH = TEST_DATA_DIR / "btc_usdt_4h_test_raw.parquet"
TEST_PROCESSED_DATA_PATH = TEST_DATA_DIR / "btc_usdt_4h_test_processed.parquet"
TEST_FEATURES_PATH = TEST_DATA_DIR / "btc_usdt_4h_test_features.parquet"

# 테스트 디렉토리 생성
os.makedirs(TEST_DATA_DIR, exist_ok=True)

def test_end_to_end_pipeline():
    """
    데이터 수집부터 백테스트까지 전체 파이프라인을 테스트합니다.
    """
    # 1. 데이터 수집
    print("1. 데이터 수집 중...")
    df = fetch_historical_data(
        symbol=TEST_SYMBOL,
        timeframe=TEST_TIMEFRAME,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        save_path=TEST_RAW_DATA_PATH
    )
    assert not df.empty, "데이터 수집 실패: 빈 데이터프레임이 반환되었습니다."
    assert os.path.exists(TEST_RAW_DATA_PATH), "원본 데이터 파일이 저장되지 않았습니다."
    print(f"   - {len(df)}개의 캔들 데이터 수집 완료")

    # 2. 데이터 품질 감사
    print("2. 데이터 품질 감사 중...")
    audit_report = run_data_audit(df, symbol=TEST_SYMBOL)
    assert "missing_values" in audit_report, "데이터 감사 보고서에 결측치 정보가 없습니다."
    print(f"   - 감사 완료: {audit_report['missing_values']}개의 결측치 발견")

    # 3. 데이터 정제
    print("3. 데이터 정제 중...")
    cleaned_df = clean_ohlcv_data(df, save_path=TEST_PROCESSED_DATA_PATH)
    assert not cleaned_df.isnull().any().any(), "정제된 데이터에 결측치가 있습니다."
    assert os.path.exists(TEST_PROCESSED_DATA_PATH), "정제된 데이터 파일이 저장되지 않았습니다."
    print(f"   - {len(cleaned_df)}개의 캔들 데이터 정제 완료")

    # 4. 피처 엔지니어링
    print("4. 피처 엔지니어링 중...")
    features_df = generate_features(cleaned_df, save_path=TEST_FEATURES_PATH)
    assert len(features_df.columns) > len(cleaned_df.columns), "피처가 제대로 생성되지 않았습니다."
    assert os.path.exists(TEST_FEATURES_PATH), "피처 데이터 파일이 저장되지 않았습니다."
    print(f"   - {len(features_df.columns) - len(cleaned_df.columns)}개의 피처 생성 완료")

    # 5. 모델 로드 및 예측
    print("5. 모델 로드 및 예측 중...")
    model_path = MODELS_DIR / "lgbm_model.pkl"
    if not model_path.exists():
        pytest.skip("훈련된 모델이 없습니다. 먼저 모델을 훈련시켜주세요.")
    
    model = load_model(model_path)
    predictions = predict(model, features_df)
    assert len(predictions) == len(features_df), "예측 결과의 길이가 입력 데이터와 일치하지 않습니다."
    print(f"   - {len(predictions)}개의 예측 완료")

    # 6. 백테스트 실행
    print("6. 백테스트 실행 중...")
    # 예측 결과를 features_df에 추가
    features_df['signal'] = predictions
    
    # 백테스트 엔진 초기화
    initial_capital = 10000.0  # 초기 자본 10,000 USDT
    commission = 0.0004  # 0.04% 수수료
    
    engine = BacktestEngine(
        data=features_df,
        initial_capital=initial_capital,
        commission=commission,
        leverage=3.0,  # 최대 3배 레버리지
        risk_per_trade=0.02  # 거래당 2% 리스크
    )
    
    # 백테스트 실행
    results = engine.run()
    
    # 결과 검증
    assert 'total_return' in results, "백테스트 결과에 수익률 정보가 없습니다."
    assert 'max_drawdown' in results, "백테스트 결과에 최대 낙폭 정보가 없습니다."
    
    print("\n=== 백테스트 결과 ===")
    print(f"기간: {TEST_START_DATE} ~ {TEST_END_DATE}")
    print(f"초기 자본: {initial_capital:,.2f} USDT")
    print(f"최종 자본: {results['final_balance']:,.2f} USDT")
    print(f"총 수익률: {results['total_return']:.2%}")
    print(f"최대 낙폭: {results['max_drawdown']:.2%}")
    print(f"샤프 지수: {results.get('sharpe_ratio', 'N/A'):.2f}")
    print(f"총 거래 횟수: {results.get('total_trades', 0)}회")
    print(f"승률: {results.get('win_rate', 0):.2%}")
    print("==================")

if __name__ == "__main__":
    # 직접 실행 시 테스트 수행
    test_end_to_end_pipeline()
