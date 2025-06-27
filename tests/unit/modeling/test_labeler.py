"""
TripleBarrierLabeler 클래스에 대한 단위 테스트
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 테스트를 위해 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.modeling.labeler import TripleBarrierLabeler, TripleBarrierParams

@pytest.fixture
def sample_price_data():
    """테스트용 샘플 가격 데이터 생성"""
    np.random.seed(42)
    n = 100
    
    # 기본 가격 생성 (랜덤 워크)
    returns = np.random.normal(0.001, 0.01, n)
    prices = 100 * np.cumprod(1 + returns)
    
    # 고가, 저가 생성 (종가 주변으로 랜덤 변동)
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    
    # 날짜 인덱스 생성
    dates = pd.date_range(start='2023-01-01', periods=n, freq='4H')
    
    # 매수 신호 생성 (랜덤)
    signals = np.zeros(n)
    signal_indices = np.random.choice(n, size=5, replace=False)  # 5개의 랜덤한 지점에 매수 신호 생성
    signals[signal_indices] = 1
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'open': prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': np.random.randint(100, 1000, n),
        'signal': signals
    }, index=dates)
    
    return df

def test_triple_barrier_labeler_initialization():
    """TripleBarrierLabeler 초기화 테스트"""
    params = TripleBarrierParams(
        take_profit=1.5,
        stop_loss=1.0,
        max_holding_period=24,
        volatility_window=20,
        volatility_scale=2.0
    )
    
    labeler = TripleBarrierLabeler(params)
    
    assert labeler.params.take_profit == 1.5
    assert labeler.params.stop_loss == 1.0
    assert labeler.params.max_holding_period == 24
    assert labeler.params.volatility_window == 20
    assert labeler.params.volatility_scale == 2.0

def test_generate_labels_with_sample_data(sample_price_data):
    """샘플 데이터로 레이블 생성 테스트"""
    # 테스트 파라미터 설정
    params = TripleBarrierParams(
        take_profit=1.5,
        stop_loss=1.0,
        max_holding_period=24,  # 24봉 (4시간 * 24 = 96시간 = 4일)
        volatility_window=20,
        volatility_scale=2.0
    )
    
    # 레이블 생성
    labeler = TripleBarrierLabeler(params)
    data = sample_price_data
    labels = labeler.generate_labels(data, signal_col='signal', price_col='close')
    
    # 검증
    assert len(labels) == len(data), "레이블 개수가 입력 데이터 길이와 일치해야 함"
    assert labels.dtype == np.int64, "레이블은 정수형이어야 함"
    
    # 매수 신호가 있는 지점 확인
    signal_indices = np.where(data['signal'] == 1)[0]
    assert len(signal_indices) > 0, "테스트 데이터에 매수 신호가 없음"
    
    # 매수 신호 이후에 레이블이 할당되었는지 확인
    for i in signal_indices:
        # 최대 보유 기간 내에서 레이블이 할당되었는지 확인
        end_idx = min(i + params.max_holding_period + 1, len(data))
        signal_labels = labels[i:end_idx]
        
        # 매수 신호 이후에 0이 아닌 레이블이 있는지 확인 (수익 실현 또는 손절)
        assert any(signal_labels != 0), f"매수 신호 {i} 이후에 레이블이 할당되지 않음"

def test_no_signal_case(sample_price_data):
    """매수 신호가 없는 경우 테스트"""
    params = TripleBarrierParams(
        take_profit=1.5,
        stop_loss=1.0,
        max_holding_period=24,
        volatility_window=20,
        volatility_scale=2.0
    )
    
    labeler = TripleBarrierLabeler(params)
    data = sample_price_data.copy()
    data['signal'] = 0  # 모든 신호를 0으로 설정
    
    labels = labeler.generate_labels(data, signal_col='signal', price_col='close')
    
    # 매수 신호가 없으므로 모든 레이블이 0이어야 함
    assert all(labels == 0), "매수 신호가 없을 때는 모든 레이블이 0이어야 함"

def test_all_signals_case(sample_price_data):
    """모든 지점이 매수 신호인 경우 테스트"""
    params = TripleBarrierParams(
        take_profit=1.5,
        stop_loss=1.0,
        max_holding_period=6,  # 짧은 보유 기간으로 설정
        volatility_window=20,
        volatility_scale=2.0
    )
    
    labeler = TripleBarrierLabeler(params)
    data = sample_price_data.copy()
    data['signal'] = 1  # 모든 지점을 매수 신호로 설정
    
    labels = labeler.generate_labels(data, signal_col='signal', price_col='close')
    
    # 매수 신호가 연속적이므로, 일부는 레이블이 0일 수 있음 (최대 보유 기간 내에 목표에 도달하지 못한 경우)
    assert len(labels) == len(data), "레이블 개수가 입력 데이터 길이와 일치해야 함"
    
    # 마지막 max_holding_period 개의 데이터 포인트는 레이블이 0일 수 있음 (아직 보유 중)
    assert any(labels != 0), "일부 레이블은 0이 아니어야 함"

if __name__ == "__main__":
    # pytest를 사용하지 않고 직접 테스트 실행 (디버깅용)
    data = sample_price_data()
    
    # 테스트 파라미터 설정
    params = TripleBarrierParams(
        take_profit=1.5,
        stop_loss=1.0,
        max_holding_period=24,
        volatility_window=20,
        volatility_scale=2.0
    )
    
    # 레이블 생성
    labeler = TripleBarrierLabeler(params)
    labels = labeler.generate_labels(data, signal_col='signal', price_col='close')
    
    # 결과 출력
    print("\n=== 테스트 결과 ===")
    print(f"데이터 길이: {len(data)}")
    print(f"매수 신호 수: {data['signal'].sum()}")
    print(f"할당된 레이블 수: {(labels != 0).sum()}")
    print("\n레이블 분포:")
    print(pd.Series(labels).value_counts().to_string())
