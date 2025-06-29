"""
모델 예측을 사용한 암호화폐 트레이딩 전략 백테스트를 위한 예제 스크립트

이 스크립트는 다음과 같은 내용을 보여줍니다:
1. OHLCV 데이터와 모델 예측값 로드
2. 백테스트 구성 및 실행
3. 결과 분석 및 시각화
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 프로젝트 루트 경로를 시스템 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, TripleBarrierStrategy

# 설정
DATA_DIR = Path('data/processed')  # 데이터 디렉토리
REPORT_DIR = Path('reports')       # 보고서 저장 디렉토리
PLOT_DIR = Path('plots')           # 차트 저장 디렉토리

# 필요한 디렉토리가 없으면 생성
for directory in [REPORT_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def load_sample_data():
    """샘플 OHLCV 데이터와 예측값을 생성합니다."""
    np.random.seed(42)  # 재현성을 위한 시드 설정
    dates = pd.date_range(start='2023-01-01', periods=500, freq='4h')
    n = len(dates)
    
    # 가상의 가격 데이터 생성
    prices = 50000 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    
    # OHLCV 데이터 생성
    data = pd.DataFrame()
    data['datetime'] = dates
    data['open'] = prices
    data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    data['close'] = prices * (1 + np.random.normal(0, 0.002, n))
    data['volume'] = np.random.randint(100, 1000, n)
    
    # 가상의 예측값 추가
    data['prediction'] = np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    
    return data

def run_backtest():
    """샘플 데이터로 백테스트를 실행하고 결과를 반환합니다."""
    engine = None
    results = None
    temp_file = None
    try:
        print("1. 샘플 데이터 생성 중...")
        data = load_sample_data()
        
        temp_file = DATA_DIR / 'temp_backtest_data.parquet'
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.to_parquet(temp_file)
        
        print("2. 백테스트 엔진 초기화 중...")
        engine = BacktestEngine(
            initial_capital=10000.0,
            commission=0.0004,
            leverage=3,
            slippage=0.0005
        )
        
        print("3. 백테스트 엔진에 데이터 로드 중...")
        engine.load_data(data)
        
        print("4. 트리플 배리어 전략 추가 중...")
        engine.add_strategy(TripleBarrierStrategy)
        
        print("5. 성과 분석기 추가 중...")
        engine.add_analyzers()
        
        print("6. 백테스트 실행 중...")
        results = engine.run_backtest()
        
        if results:
            print("\n=== 백테스트 결과 요약 ===")
            print(f"초기 자본: {results.get('initial_capital', 0):,.2f} USDT")
            print(f"최종 자산: {results.get('final_value', 0):,.2f} USDT")
            print(f"총 수익률: {results.get('return_pct', 0):.2f}%")
            
            kpis = results.get('kpis', {})
            if kpis:
                print("\n=== 주요 지표 ===")
                print(f"샤프 지수: {kpis.get('sharpe_ratio', 0):.2f}")
                print(f"최대 낙폭(MDD): {kpis.get('max_drawdown_pct', 0):.2f}%")
                print(f"총 거래 횟수: {kpis.get('number_of_trades', 0)}")
                print(f"승률: {kpis.get('win_rate', 0):.2f}%")
                print(f"수익 팩터: {kpis.get('profit_factor', 0):.2f}")
                print(f"SQN: {kpis.get('sqn', 0):.2f}")
        else:
            print("\n경고: 백테스트 결과가 반환되지 않았습니다.")

    except Exception as e:
        print(f"\n백테스트 실행 중 심각한 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 임시 데이터 파일 삭제
        if temp_file and temp_file.exists():
            temp_file.unlink()
            print(f"\n임시 파일 삭제: {temp_file}")

    return engine, results

def plot_equity_curve(engine, save_path=None):
    """백테스트 결과로부터 자산 곡선을 그립니다."""
    if not isinstance(engine, BacktestEngine) or not hasattr(engine, 'results') or not engine.results:
        print("\n경고: 유효한 백테스트 엔진 또는 결과가 없어 자산 곡선을 그릴 수 없습니다.")
        return
        
    try:
        equity_data = engine.results.get('equity_curve')
        
        # equity_curve가 딕셔너리인 경우 pandas Series로 변환
        if isinstance(equity_data, dict):
            if not equity_data:  # 빈 딕셔너리인 경우
                print("경고: 자산 곡선 데이터가 비어 있습니다.")
                return
            # 딕셔너리를 pandas Series로 변환 (날짜 인덱스가 있는 경우)
            equity_curve = pd.Series(equity_data)
            if not isinstance(equity_curve.index, pd.DatetimeIndex):
                # 인덱스가 날짜 형식이 아닌 경우, 단순 정수 인덱스 사용
                equity_curve.index = pd.RangeIndex(len(equity_curve))
        elif hasattr(equity_data, 'empty') and equity_data.empty:
            print("경고: 자산 곡선 데이터가 비어 있습니다.")
            return
        else:
            equity_curve = equity_data

        plt.figure(figsize=(12, 6))
        
        # 시계열 데이터 플로팅 (날짜 인덱스가 있는 경우)
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            plt.plot(equity_curve.index, equity_curve, label='Equity Curve')
            plt.gcf().autofmt_xdate()  # 날짜 레이블 자동 조정
        else:
            # 날짜 인덱스가 없는 경우 단순 인덱스 사용
            plt.plot(equity_curve.values, label='Equity Curve')
            
        plt.title('Backtest Equity Curve')
        plt.xlabel('Date' if isinstance(equity_curve.index, pd.DatetimeIndex) else 'Period')
        plt.ylabel('Equity (USDT)')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            PLOT_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"자산 곡선이 저장되었습니다: {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"\n자산 곡선을 그리는 중 오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("백테스트 예제를 시작합니다...")
    engine, results = run_backtest()

    if results:
        plot_path = PLOT_DIR / 'equity_curve.png'
        print(f"\n자산 곡선을 그리는 중: {plot_path}")
        plot_equity_curve(engine, save_path=str(plot_path))
        print("\n백테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n백테스트 실행에 실패했거나 결과가 없습니다.")
