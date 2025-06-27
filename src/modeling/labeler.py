"""
타겟 레이블링 모듈 (labeler.py)

삼중 장벽 기법(Triple Barrier Method)을 사용하여 시계열 데이터에 대한 레이블을 생성합니다.
이 모듈은 금융 시계열 데이터에 대해 수익 실현, 손절, 시간 초과에 따른 레이블을 생성합니다.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TripleBarrierParams:
    """삼중 장벽 기법을 위한 파라미터 클래스"""
    take_profit: float = 1.5  # 수익 목표 (손절 대비 배수)
    stop_loss: float = 1.0    # 손절 기준 (ATR 배수)
    max_holding_period: int = 24  # 최대 보유 기간 (4시간 봉 기준 24봉 = 96시간 = 4일)
    volatility_window: int = 20  # 변동성(ATR) 계산 기간
    volatility_scale: float = 2.0  # ATR 배수 (손절 수준 조정)

class TripleBarrierLabeler:
    """
    삼중 장벽 기법을 사용하여 시계열 데이터에 레이블을 생성하는 클래스
    
    이 클래스는 금융 시계열 데이터에 대해 수익 실현, 손절, 시간 초과에 따른
    레이블(1, -1, 0)을 생성합니다.
    """
    
    def __init__(self, params: Optional[TripleBarrierParams] = None):
        """
        TripleBarrierLabeler 초기화
        
        Args:
            params: 삼중 장벽 기법 파라미터 (기본값 사용 시 None)
        """
        self.params = params if params is not None else TripleBarrierParams()
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     window: int = None) -> pd.Series:
        """
        ATR(Average True Range) 계산
        
        Args:
            high: 고가 시리즈
            low: 저가 시리즈
            close: 종가 시리즈
            window: ATR 계산 기간 (None이면 params 사용)
            
        Returns:
            pd.Series: ATR 값
        """
        window = window or self.params.volatility_window
        
        try:
            # pandas-ta 라이브러리 사용 시도
            import pandas_ta as ta
            atr = ta.atr(high=high, low=low, close=close, length=window)
            if atr is not None and not atr.empty:
                return atr
        except (ImportError, Exception) as e:
            logger.warning(f"pandas-ta ATR 계산 실패: {e}")
        
        # pandas-ta 실패 시 수동 계산
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    def generate_labels(self, data: pd.DataFrame, 
                       signal_col: str = 'signal',
                       price_col: str = 'close') -> pd.Series:
        """
        삼중 장벽 기법을 사용하여 레이블 생성
        
        Args:
            data: OHLCV 데이터를 포함한 DataFrame
            signal_col: 매수 신호 컬럼 (1: 매수, 0: 무시)
            price_col: 가격 컬럼 (기본값: 'close')
            
        Returns:
            pd.Series: 각 시점의 레이블 (1: 수익 실현, -1: 손절, 0: 시간 초과/무관심)
        """
        if signal_col not in data.columns:
            raise ValueError(f"신호 컬럼 '{signal_col}'을(를) 찾을 수 없습니다.")
            
        if price_col not in data.columns:
            raise ValueError(f"가격 컬럼 '{price_col}'을(를) 찾을 수 없습니다.")
        
        # ATR 계산 (변동성 기반 손절 수준 설정용)
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            atr = self.calculate_atr(
                data['high'], 
                data['low'], 
                data['close'],
                window=self.params.volatility_window
            )
        else:
            # 고가/저가 정보가 없을 경우 종가의 변동성 사용
            returns = data[price_col].pct_change()
            atr = returns.rolling(window=self.params.volatility_window).std() * data[price_col]
        
        # 레이블 초기화 (기본값: 0 - 무관심/보유)
        labels = pd.Series(0, index=data.index, name='label')
        
        # 매수 신호가 발생한 지점 찾기
        buy_signals = data[data[signal_col] == 1].index
        
        for signal_time in buy_signals:
            # 신호 발생 시점 인덱스
            signal_idx = data.index.get_loc(signal_time)
            
            # 최대 보유 기간 내의 데이터 선택
            max_idx = min(signal_idx + self.params.max_holding_period + 1, len(data))
            future_data = data.iloc[signal_idx:max_idx]
            
            if future_data.empty:
                continue
                
            # 진입 가격 및 장벽 설정
            entry_price = future_data[price_col].iloc[0]
            atr_value = atr.loc[signal_time] if signal_time in atr.index else atr.iloc[signal_idx] if signal_idx < len(atr) else 0
            
            # 수익 목표 및 손절 수준 (ATR 기반)
            stop_loss_level = entry_price - (atr_value * self.params.volatility_scale)
            take_profit_level = entry_price + (atr_value * self.params.volatility_scale * self.params.take_profit)
            
            # 미래 가격 추적
            future_prices = future_data[price_col]
            
            # 각 시점에서 이벤트 확인
            for i, (time, price) in enumerate(future_prices.items()):
                # 수익 실현 (Take Profit)
                if price >= take_profit_level:
                    labels.loc[time] = 1  # 수익 실현
                    break
                # 손절 (Stop Loss)
                elif price <= stop_loss_level:
                    labels.loc[time] = -1  # 손절
                    break
                # 시간 초과 (마지막 봉)
                elif i == len(future_prices) - 1:
                    labels.loc[time] = 0  # 시간 초과 (횡보)
        
        return labels

def create_labels(data: pd.DataFrame, signal_col: str = 'signal', 
                 price_col: str = 'close',
                 params: Optional[TripleBarrierParams] = None) -> pd.Series:
    """
    삼중 장벽 기법을 사용하여 레이블을 생성하는 편의 함수
    
    Args:
        data: OHLCV 데이터를 포함한 DataFrame
        signal_col: 매수 신호 컬럼 (1: 매수, 0: 무시)
        price_col: 가격 컬럼 (기본값: 'close')
        params: 삼중 장벽 기법 파라미터 (기본값 사용 시 None)
        
    Returns:
        pd.Series: 각 시점의 레이블 (1: 수익 실현, -1: 손절, 0: 시간 초과/무관심)
    """
    labeler = TripleBarrierLabeler(params)
    return labeler.generate_labels(data, signal_col, price_col)
