"""
실제 거래소 데이터로 레이블러 테스트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os
import ccxt
from datetime import datetime, timedelta

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 현재 스크립트의 부모 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath('.'))

# 레이블러 임포트
from src.modeling.labeler import TripleBarrierLabeler, TripleBarrierParams

def fetch_binance_data(symbol='BTC/USDT', timeframe='4h', days=30):
    """바이낸스에서 과거 데이터 가져오기"""
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # 현재 시간부터 days일 전까지의 데이터 가져오기
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    try:
        # 바이낸스에서 OHLCV 데이터 가져오기
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        
        # 데이터프레임으로 변환
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 간단한 매수 신호 생성 (랜덤, 실제로는 전략에 따라 결정)
        df['signal'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
        
        return df
    except Exception as e:
        print(f"데이터 가져오기 오류: {e}")
        return None

def main():
    # 바이낸스에서 BTC/USDT 4시간봉 데이터 가져오기
    print("바이낸스에서 BTC/USDT 4시간봉 데이터를 가져오는 중...")
    data = fetch_binance_data(days=60)  # 최근 60일 데이터
    
    if data is None or data.empty:
        print("데이터를 가져오는데 실패했습니다.")
        return
    
    print(f"가져온 데이터 기간: {data.index[0]} ~ {data.index[-1]}")
    print(f"총 데이터 포인트: {len(data)}")
    
    # 삼중 장벽 파라미터 설정
    params = TripleBarrierParams(
        take_profit=1.5,  # 수익 목표: 손절 대비 1.5배
        stop_loss=1.0,     # 손절 기준: ATR 1배
        max_holding_period=24 * 7,  # 최대 보유 기간: 24봉 * 7 = 1주일 (4시간봉 기준)
        volatility_window=20,       # ATR 계산 기간
        volatility_scale=2.0        # ATR 배수 (손절 수준 조정)
    )
    
    # 레이블 생성
    print("\n삼중 장벽 기법으로 레이블 생성 중...")
    labeler = TripleBarrierLabeler(params)
    labels = labeler.generate_labels(data, signal_col='signal', price_col='close')
    
    # 결과 병합
    result = data.copy()
    result['label'] = labels
    
    # 신호 및 레이블이 있는 지점 필터링
    signal_points = result[result['signal'] == 1]
    labeled_points = result[result['label'] != 0]
    
    # 시각화
    plt.figure(figsize=(15, 8))
    
    # 가격 차트
    plt.plot(result.index, result['close'], label='Price', alpha=0.7, linewidth=2, color='#1f77b4')
    
    # 매수 신호 표시 (초록색 위쪽 삼각형)
    if not signal_points.empty:
        plt.scatter(
            signal_points.index, 
            signal_points['close'], 
            color='lime', 
            marker='^', 
            s=150,
            label='Buy Signal',
            alpha=0.9,
            edgecolors='darkgreen',
            linewidths=1.5
        )
    
    # 레이블 표시
    if not labeled_points.empty:
        # 수익 실현 (파란색 동그라미)
        tp_points = labeled_points[labeled_points['label'] == 1]
        if not tp_points.empty:
            plt.scatter(
                tp_points.index, 
                tp_points['close'], 
                color='blue', 
                marker='o', 
                s=150,
                label='Take Profit',
                alpha=0.9,
                edgecolors='darkblue',
                linewidths=1.5
            )
        
        # 손절 (빨간색 X)
        sl_points = labeled_points[labeled_points['label'] == -1]
        if not sl_points.empty:
            plt.scatter(
                sl_points.index, 
                sl_points['close'], 
                color='red', 
                marker='x', 
                s=150,
                label='Stop Loss',
                alpha=0.9,
                linewidths=2
            )
    
    plt.title('BTC/USDT 4h - Triple Barrier Labeling', fontsize=15, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USDT)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # x축 레이블 회전
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 이미지 저장
    output_path = 'btc_usdt_triple_barrier.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n시각화 결과를 '{output_path}' 파일로 저장했습니다.")
    
    # 결과 요약 출력
    print("\n=== 레이블링 결과 요약 ===")
    print(f"데이터 기간: {result.index[0].strftime('%Y-%m-%d')} ~ {result.index[-1].strftime('%Y-%m-%d')}")
    print(f"총 데이터 포인트: {len(result)}")
    print(f"매수 신호 수: {len(signal_points)}")
    print(f"레이블이 할당된 포인트 수: {len(labeled_points)}")
    
    if not labeled_points.empty:
        print("\n레이블 분포:")
        label_counts = labeled_points['label'].value_counts()
        for label, count in label_counts.items():
            if label == 1:
                print(f"- 수익 실현 (Take Profit): {count}건")
            elif label == -1:
                print(f"- 손절 (Stop Loss): {count}건")
    
    # 그래프 표시
    plt.show()

if __name__ == "__main__":
    main()
