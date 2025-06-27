"""
타겟 변수 생성 스크립트

이 스크립트는 피처가 추가된 데이터에 삼중 장벽 기법을 적용하여 타겟 변수를 생성합니다.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modeling.labeler import TripleBarrierLabeler, create_labels, TripleBarrierParams

def generate_signals(data: pd.DataFrame) -> pd.Series:
    """
    매수/매도 신호를 생성합니다.
    여기서는 단순히 모든 시점에서 매수하는 전략을 사용합니다.
    실제로는 더 복잡한 전략을 구현할 수 있습니다.
    
    Args:
        data: OHLCV 및 기술적 지표를 포함한 데이터프레임
        
    Returns:
        pd.Series: 1(매수) 또는 0(보류) 값을 가진 시리즈
    """
    # 모든 시점에서 매수한다고 가정 (1: 매수, 0: 보류)
    signals = pd.Series(1, index=data.index)
    return signals

def main():
    """메인 함수"""
    try:
        # 입력 및 출력 파일 경로 설정
        input_path = Path('data/btc_usdt_4h_features.parquet')
        output_path = Path('data/btc_usdt_4h_with_target.parquet')
        
        # 디렉토리 생성 (없는 경우)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 데이터 로드
        print(f"데이터 로드 중: {input_path}")
        data = pd.read_parquet(input_path)
        
        # 날짜 인덱스 설정 (필요한 경우)
        if 'datetime' in data.columns:
            data = data.set_index('datetime')
        
        # 매수/매도 신호 생성
        print("매수/매도 신호 생성 중...")
        data['signal'] = generate_signals(data)
        
        # 삼중 장벽 파라미터 설정
        params = TripleBarrierParams(
            take_profit=1.5,  # 수익 목표 (손절 대비 1.5배)
            stop_loss=1.0,     # 손절 기준 (ATR 배수)
            max_holding_period=24,  # 최대 보유 기간 (4시간 봉 기준 24봉 = 96시간 = 4일)
            volatility_window=20,   # ATR 계산 기간
            volatility_scale=2.0    # ATR 배수 (손절 수준 조정)
        )
        
        # 타겟 변수 생성
        print("타겟 변수 생성 중...")
        target_series = create_labels(
            data,
            signal_col='signal',
            price_col='close',
            params=params
        )
        
        # 디버깅: 반환된 객체의 타입과 내용 확인
        print(f"반환된 객체 타입: {type(target_series)}")
        print(f"반환된 객체 길이: {len(target_series) if hasattr(target_series, '__len__') else 'N/A'}")
        print(f"반환된 객체 샘플: {target_series.head() if hasattr(target_series, 'head') else 'N/A'}")
        
        # 타겟 변수를 원본 데이터에 추가
        data_with_target = data.copy()
        if isinstance(target_series, pd.Series):
            data_with_target['target'] = target_series
        else:
            # 만약 target_series가 시리즈가 아니라면, 인덱스를 기준으로 병합
            data_with_target = data_with_target.join(pd.Series(target_series, name='target'))
        
        # 결과 저장
        data_with_target.to_parquet(output_path)
        print(f"저장된 데이터 컬럼: {data_with_target.columns.tolist()}")
        
        # 결과 요약 출력
        print("\n타겟 변수 분포:")
        print(data_with_target['target'].value_counts())
        print(f"\n타겟 변수가 추가된 데이터를 저장했습니다: {output_path}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
