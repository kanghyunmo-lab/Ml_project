from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta as ta  # 기술적 분석을 위한 라이브러리
from typing import Optional, Dict, Any
import logging

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
            sma_20 = ta.sma(self.data['close'], length=20)
            if sma_20 is not None:
                self.data['SMA_20'] = sma_20
            else:
                self.data['SMA_20'] = self.data['close'].rolling(window=20).mean()
            
            sma_50 = ta.sma(self.data['close'], length=50)
            if sma_50 is not None:
                self.data['SMA_50'] = sma_50
            else:
                self.data['SMA_50'] = self.data['close'].rolling(window=50).mean()
            
            sma_200 = ta.sma(self.data['close'], length=200)
            if sma_200 is not None:
                self.data['SMA_200'] = sma_200
            else:
                self.data['SMA_200'] = self.data['close'].rolling(window=200).mean()
            
            # 지수이동평균(EMA)
            ema_12 = ta.ema(self.data['close'], length=12)
            if ema_12 is not None:
                self.data['EMA_12'] = ema_12
            else:
                self.data['EMA_12'] = self.data['close'].ewm(span=12, adjust=False).mean()
            
            ema_26 = ta.ema(self.data['close'], length=26)
            if ema_26 is not None:
                self.data['EMA_26'] = ema_26
            else:
                self.data['EMA_26'] = self.data['close'].ewm(span=26, adjust=False).mean()
                
        except Exception as e:
            logger.warning(f"이동평균선 계산 중 오류 발생: {e}")
            # 이동평균선 계산에 실패해도 다른 지표는 계속 계산
    
    def _add_rsi(self, period: int = 14) -> None:
        """RSI(상대강도지수) 추가"""
        try:
            rsi = ta.rsi(self.data['close'], length=period)
            if rsi is not None:
                self.data[f'RSI_{period}'] = rsi
            else:
                # RSI 계산 실패 시 대체 방법으로 계산
                delta = self.data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                self.data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.warning(f"RSI 계산 중 오류 발생: {e}")
            # RSI 계산에 실패해도 다른 지표는 계속 계산
    
    def _add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        """MACD 지표 추가"""
        try:
            # pandas-ta의 MACD 함수 호출
            macd_result = ta.macd(
                close=self.data['close'], 
                fast=fast, 
                slow=slow, 
                signal=signal,
                append=False  # 결과를 원본에 추가하지 않음
            )
            
            # MACD 계산 결과가 유효한지 확인
            if macd_result is not None and not macd_result.empty:
                # 컬럼 이름 확인
                macd_col = None
                signal_col = None
                hist_col = None
                
                # 가능한 컬럼 이름 패턴
                possible_prefixes = [
                    f'MACD_{fast}_{slow}_{signal}_',  # 일부 버전
                    'MACD_',  # 다른 일반적인 접두사
                    ''        # 접두사 없음
                ]
                
                # 실제 컬럼 이름 찾기
                for prefix in possible_prefixes:
                    if f'{prefix}macd' in macd_result.columns:
                        macd_col = f'{prefix}macd'
                    if f'{prefix}signal' in macd_result.columns:
                        signal_col = f'{prefix}signal'
                    if f'{prefix}hist' in macd_result.columns:
                        hist_col = f'{prefix}hist'
                
                # 컬럼이 존재하면 추가
                if macd_col is not None:
                    self.data['MACD'] = macd_result[macd_col]
                if signal_col is not None:
                    self.data['MACD_signal'] = macd_result[signal_col]
                if hist_col is not None:
                    self.data['MACD_hist'] = macd_result[hist_col]
            else:
                # MACD 계산 실패 시 직접 계산
                self._calculate_macd_manually(fast, slow, signal)
                
        except Exception as e:
            logger.warning(f"MACD 계산 중 오류 발생: {e}")
            # MACD 계산에 실패하면 직접 계산 시도
            self._calculate_macd_manually(fast, slow, signal)
    
    def _calculate_macd_manually(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        """MACD를 수동으로 계산하는 메서드"""
        try:
            # 지수이동평균(EMA) 계산
            ema_fast = self.data['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = self.data['close'].ewm(span=slow, adjust=False).mean()
            
            # MACD 라인 (빠른 EMA - 느린 EMA)
            self.data['MACD'] = ema_fast - ema_slow
            
            # 시그널 라인 (MACD의 signal 기간 EMA)
            self.data['MACD_signal'] = self.data['MACD'].ewm(span=signal, adjust=False).mean()
            
            # MACD 히스토그램 (MACD - 시그널)
            self.data['MACD_hist'] = self.data['MACD'] - self.data['MACD_signal']
            
        except Exception as e:
            logger.warning(f"수동 MACD 계산 중 오류 발생: {e}")
    
    def _add_bollinger_bands(self, window: int = 20, num_std: int = 2) -> None:
        """볼린저 밴드 추가"""
        try:
            bbands = ta.bbands(self.data['close'], length=window, std=num_std)
            if bbands is not None and not bbands.empty:
                # 가능한 컬럼 이름 패턴
                possible_prefixes = [
                    f'BBANDS_{window}_{num_std}.',  # 일부 버전
                    'BB_',  # 다른 일반적인 접두사
                    ''      # 접두사 없음
                ]
                
                # 각 밴드에 대한 가능한 컬럼 이름 생성
                upper_cols = [f'{p}upper' for p in possible_prefixes] + [f'BBU_{window}_{num_std}']
                middle_cols = [f'{p}middle' for p in possible_prefixes] + [f'BBM_{window}_{num_std}']
                lower_cols = [f'{p}lower' for p in possible_prefixes] + [f'BBL_{window}_{num_std}']
                
                # 실제 존재하는 컬럼 찾기
                upper_col = next((col for col in upper_cols if col in bbands.columns), None)
                middle_col = next((col for col in middle_cols if col in bbands.columns), None)
                lower_col = next((col for col in lower_cols if col in bbands.columns), None)
                
                # 컬럼이 존재하면 추가
                if upper_col is not None:
                    self.data['BB_upper'] = bbands[upper_col]
                if middle_col is not None:
                    self.data['BB_middle'] = bbands[middle_col]
                if lower_col is not None:
                    self.data['BB_lower'] = bbands[lower_col]
        except Exception as e:
            logger.warning(f"볼린저 밴드 계산 중 오류 발생: {e}")
            # 볼린저 밴드 계산에 실패해도 다른 지표는 계속 계산
    
    def _add_atr(self, window: int = 14) -> None:
        """ATR(평균진폭) 추가"""
        atr = ta.atr(
            high=self.data['high'], 
            low=self.data['low'], 
            close=self.data['close'], 
            length=window
        )
        if atr is not None:
            self.data['ATR'] = atr
        else:
            # ATR 계산 실패 시 대체 방법으로 계산
            high_low = self.data['high'] - self.data['low']
            high_close = (self.data['high'] - self.data['close'].shift()).abs()
            low_close = (self.data['low'] - self.data['close'].shift()).abs()
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            self.data['ATR'] = true_range.rolling(window=window).mean()
    
    def _add_volume_indicators(self) -> None:
        """거래량 관련 지표 추가"""
        try:
            # 거래량 이동평균
            vma_20 = ta.sma(self.data['volume'], length=20)
            if vma_20 is not None:
                self.data['VMA_20'] = vma_20
            else:
                self.data['VMA_20'] = self.data['volume'].rolling(window=20).mean()
            
            # OBV(On-Balance Volume)
            obv = ta.obv(self.data['close'], self.data['volume'])
            if obv is not None:
                self.data['OBV'] = obv
            else:
                # OBV 계산 실패 시 대체 방법으로 계산
                obv = [0] * len(self.data)
                obv[0] = self.data['volume'].iloc[0] if len(self.data) > 0 else 0
                
                for i in range(1, len(self.data)):
                    if self.data['close'].iloc[i] > self.data['close'].iloc[i-1]:
                        obv[i] = obv[i-1] + self.data['volume'].iloc[i]
                    elif self.data['close'].iloc[i] < self.data['close'].iloc[i-1]:
                        obv[i] = obv[i-1] - self.data['volume'].iloc[i]
                    else:
                        obv[i] = obv[i-1]
                
                self.data['OBV'] = obv
                
        except Exception as e:
            logger.warning(f"거래량 지표 계산 중 오류 발생: {e}")
            # 거래량 지표 계산에 실패해도 다른 지표는 계속 계산


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
