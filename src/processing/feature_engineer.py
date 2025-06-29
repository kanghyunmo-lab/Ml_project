from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import logging
from datetime import timedelta

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    기술적 분석 지표를 계산하여 피처를 생성하는 클래스
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        초기화 메서드
        
        Args:
            data (pd.DataFrame): OHLCV 데이터를 포함한 DataFrame
        """
        self.data = data.copy()
        self._validate_data()
    
    def _validate_data(self) -> None:
        """
        입력 데이터의 유효성을 검증하는 메서드
        """
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(self.data.columns):
            missing = required_columns - set(self.data.columns)
            raise ValueError(f"입력 데이터에 다음 컬럼이 누락되었습니다: {missing}")
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """
        모든 기술적 지표를 추가하는 메인 메서드
        
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 DataFrame
        """
        logger.info("기술적 지표 추가를 시작합니다...")
        
        # 이동평균선 추가
        self._add_moving_averages()
        
        # 모멘텀 지표 추가
        self._add_rsi()
        self._add_macd()
        
        # 변동성 지표 추가
        self._add_bollinger_bands()
        self._add_atr()
        
        # 거래량 지표 추가
        self._add_volume_indicators()
        
        logger.info("모든 기술적 지표 추가가 완료되었습니다.")
        return self.data
    
    def _add_moving_averages(self) -> None:
        """이동평균선 추가"""
        try:
            # 단순이동평균(SMA)
            self.data['SMA_20'] = self.data['close'].rolling(window=20).mean()
            self.data['SMA_50'] = self.data['close'].rolling(window=50).mean()
            self.data['SMA_200'] = self.data['close'].rolling(window=200).mean()
            
            # 지수이동평균(EMA)
            self.data['EMA_12'] = self.data['close'].ewm(span=12, adjust=False).mean()
            self.data['EMA_26'] = self.data['close'].ewm(span=26, adjust=False).mean()
                
        except Exception as e:
            logger.warning(f"이동평균선 계산 중 오류 발생: {e}")
            # 이동평균선 계산에 실패해도 다른 지표는 계속 계산
    
    def _calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI(Relative Strength Index) 계산
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _add_rsi(self, period: int = 14) -> None:
        """
        RSI(Relative Strength Index) 추가
        """
        self.data[f'rsi_{period}'] = self._calculate_rsi(self.data['close'], period)
    
    def _calculate_ema(self, data: pd.Series, span: int) -> pd.Series:
        """지수이동평균(EMA) 계산"""
        return data.ewm(span=span, adjust=False).mean()
    
    def _add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        """
        MACD(Moving Average Convergence Divergence) 추가
        """
        close = self.data['close']
        exp1 = close.ewm(span=fast, adjust=False).mean()
        exp2 = close.ewm(span=slow, adjust=False).mean()
        
        self.data['macd'] = exp1 - exp2
        self.data['macd_signal'] = self.data['macd'].ewm(span=signal, adjust=False).mean()
        self.data['macd_hist'] = self.data['macd'] - self.data['macd_signal']
    
    def _add_bollinger_bands(self, window: int = 20, window_dev: float = 2.0) -> None:
        """
        볼린저 밴드 추가
        """
        close = self.data['close']
        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        
        self.data['bb_upper'] = sma + (std * window_dev)
        self.data['bb_middle'] = sma
        self.data['bb_lower'] = sma - (std * window_dev)
        self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """ATR(Average True Range) 계산"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    def _add_atr(self, window: int = 14) -> None:
        """
        ATR(Average True Range) 추가
        """
        self.data['atr'] = self._calculate_atr(
            self.data['high'], 
            self.data['low'], 
            self.data['close'], 
            window
        )
    
    def _add_volume_indicators(self) -> None:
        """
        거래량 기반 지표 추가
        """
        try:
            # 거래량 이동평균
            self.data['volume_ma_20'] = self.data['volume'].rolling(window=20).mean()
            
            # 거래량 모멘텀 (전일 대비 변화율)
            self.data['volume_momentum'] = self.data['volume'].pct_change()
            
            # OBV (On-Balance Volume)
            obv = (np.sign(self.data['close'].diff()) * self.data['volume']).fillna(0).cumsum()
            self.data['obv'] = obv
            self.data['obv_ema'] = self._calculate_ema(obv, 20)
            
            # 거래량 가격 추세 (Volume Price Trend)
            self.data['vpt'] = (self.data['volume'] * 
                              ((self.data['close'] - self.data['close'].shift(1)) / 
                               self.data['close'].shift(1))).fillna(0).cumsum()
            
            # 거래량 변동성
            self.data['volume_volatility'] = self.data['volume'].rolling(window=20).std() / \
                                           self.data['volume'].rolling(window=20).mean()
            
        except Exception as e:
            logger.error(f"거래량 지표 계산 중 오류 발생: {e}")
            raise


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    FeatureEngineer를 사용하여 피처를 생성하는 편의 함수
    
    Args:
        data (pd.DataFrame): OHLCV 데이터를 포함한 DataFrame
        
    Returns:
        pd.DataFrame: 피처가 추가된 DataFrame
    """
    fe = FeatureEngineer(data)
    return fe.add_technical_indicators()


def generate_features(data: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    피처를 생성하고 선택적으로 저장하는 함수
    
    Args:
        data (pd.DataFrame): 입력 데이터프레임 (OHLCV 데이터 포함)
        save_path (Optional[str]): 피처를 저장할 파일 경로. None인 경우 저장하지 않음.
        
    Returns:
        pd.DataFrame: 피처가 추가된 데이터프레임
    """
    try:
        logger.info("피처 엔지니어링을 시작합니다...")
        
        # FeatureEngineer를 사용하여 피처 생성
        fe = FeatureEngineer(data)
        features_df = fe.add_technical_indicators()
        
        # 결과 저장
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일 확장자에 따라 저장 포맷 결정
            if save_path.suffix == '.parquet':
                features_df.to_parquet(save_path, index=False)
            elif save_path.suffix == '.csv':
                features_df.to_csv(save_path, index=False)
            else:
                # 기본값으로 parquet 형식 사용
                save_path = save_path.with_suffix('.parquet')
                features_df.to_parquet(save_path, index=False)
            
            logger.info(f"피처가 저장되었습니다: {save_path}")
        
        logger.info("피처 엔지니어링이 완료되었습니다.")
        return features_df
        
    except Exception as e:
        logger.error(f"피처 엔지니어링 중 오류가 발생했습니다: {str(e)}")
        raise


if __name__ == "__main__":
    # 테스트 코드
    import yfinance as yf
    
    # 테스트 데이터 다운로드 (예시로 BTC-USD 데이터 사용)
    btc_data = yf.download('BTC-USD', start='2020-01-01', end='2023-01-01', interval='1d')
    btc_data = btc_data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # 피처 엔지니어링 실행
    features = create_features(btc_data)
    print(f"생성된 피처 수: {len(features.columns) - 5}")
    print("추가된 피처 목록:", [col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
