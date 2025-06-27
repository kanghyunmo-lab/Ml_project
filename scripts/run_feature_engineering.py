"""
피처 엔지니어링 실행 스크립트

이 스크립트는 정제된 OHLCV 데이터에 기술적 지표를 추가하여 피처를 생성합니다.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processing.feature_engineer import FeatureEngineer, generate_features

def main():
    """메인 함수"""
    try:
        # 입력 및 출력 파일 경로 설정
        input_path = Path('data/processed/btc_usdt_4h_20191231_20250627_cleaned.parquet')
        output_path = Path('data/btc_usdt_4h_features.parquet')
        
        # 디렉토리 생성 (없는 경우)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 데이터 로드
        print(f"데이터 로드 중: {input_path}")
        data = pd.read_parquet(input_path)
        
        # 피처 생성
        print("피처 생성 중...")
        data_with_features = generate_features(data, save_path=str(output_path))
        
        # 결과 출력
        print("\n생성된 피처 목록:")
        print(list(set(data_with_features.columns) - set(data.columns)))
        print(f"\n피처가 추가된 데이터를 저장했습니다: {output_path}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
