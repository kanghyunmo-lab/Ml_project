import pandas as pd
import numpy as np
from src.processing.feature_engineer import FeatureEngineer

# 간단한 테스트 데이터 생성
data = {
    'open': [100, 101, 102, 103, 104],
    'high': [101, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [100, 102, 103, 104, 105],
    'volume': [1000, 1200, 1100, 1300, 1400]
}
df = pd.DataFrame(data)

# FeatureEngineer 인스턴스 생성
engineer = FeatureEngineer(df)

# 기술적 지표 추가
features = engineer.add_technical_indicators()

# 결과 출력
print("=== 원본 데이터 ===")
print(df)

print("\n=== 피처가 추가된 데이터 ===")
print(features.columns.tolist())

# 결측치 확인
print("\n=== 결측치 개수 ===")
print(features.isnull().sum())
