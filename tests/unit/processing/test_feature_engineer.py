import pandas as pd
import numpy as np
from src.processing.feature_engineer import FeatureEngineer

def generate_sample_data(days=100):
    """테스트용 샘플 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
    
    # 랜덤 워크를 사용한 가격 생성
    returns = np.random.normal(0.001, 0.02, days)
    prices = 50000 * (1 + returns).cumprod()
    
    # 고가, 저가, 종가 생성
    high = prices * (1 + np.abs(np.random.normal(0.005, 0.01, days)))
    low = prices * (1 - np.abs(np.random.normal(0.005, 0.01, days)))
    close = (high + low) / 2 + np.random.normal(0, 100, days)
    
    # 거래량 생성
    volume = np.random.lognormal(10, 1, days).astype(int)
    
    data = pd.DataFrame({
        'open': prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return data

def test_feature_engineer():
    # 테스트 데이터 생성
    print("테스트 데이터 생성 중...")
    sample_data = generate_sample_data(200)
    
    # FeatureEngineer 인스턴스 생성
    print("\nFeatureEngineer 초기화 중...")
    engineer = FeatureEngineer(sample_data)
    
    # 기술적 지표 추가
    print("\n기술적 지표 추가 중...")
    features = engineer.add_technical_indicators()
    
    # 결과 출력
    print("\n=== 피처 엔지니어링 결과 ===")
    print(f"원본 컬럼 수: {len(sample_data.columns)}")
    print(f"피처 추가 후 컬럼 수: {len(features.columns)}")
    print("\n추가된 피처 목록:")
    for col in features.columns.difference(['open', 'high', 'low', 'close', 'volume']):
        print(f"- {col}")
    
    print("\n=== 샘플 데이터 (처음 5행) ===")
    print(features.head())
    
    print("\n=== 결측치 확인 ===")
    print(features.isnull().sum())
    
    return features

if __name__ == "__main__":
    test_feature_engineer()
