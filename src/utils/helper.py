"""
유틸리티 함수 모음
"""
import os
import logging
from pathlib import Path
from typing import List, Union, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """로깅 설정"""
    import logging.config
    from src.config import LOG_CONFIG
    
    # 로그 디렉토리 생성
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 로깅 설정 적용
    logging.config.dictConfig(LOG_CONFIG)

def create_directories(directories: List[Union[str, Path]]) -> None:
    """필요한 디렉토리들을 생성합니다.
    
    Args:
        directories: 생성할 디렉토리 경로 리스트
    """
    for directory in directories:
        if isinstance(directory, str):
            directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

def save_dataframe(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> bool:
    """데이터프레임을 파일로 저장합니다.
    
    Args:
        df: 저장할 데이터프레임
        path: 저장 경로
        **kwargs: pandas.to_* 함수에 전달할 추가 인자
        
    Returns:
        bool: 저장 성공 여부
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.parquet':
            df.to_parquet(path, **kwargs)
        elif path.suffix == '.csv':
            df.to_csv(path, **kwargs)
        elif path.suffix == '.pkl':
            df.to_pickle(path, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")
            
        logger.info(f"데이터가 성공적으로 저장되었습니다: {path}")
        return True
    except Exception as e:
        logger.error(f"데이터 저장 중 오류가 발생했습니다: {e}")
        return False

def load_dataframe(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """파일로부터 데이터프레임을 로드합니다.
    
    Args:
        path: 로드할 파일 경로
        **kwargs: pandas.read_* 함수에 전달할 추가 인자
        
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    try:
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
            
        if path.suffix == '.parquet':
            df = pd.read_parquet(path, **kwargs)
        elif path.suffix == '.csv':
            df = pd.read_csv(path, **kwargs)
        elif path.suffix == '.pkl':
            df = pd.read_pickle(path, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")
            
        logger.info(f"데이터가 성공적으로 로드되었습니다: {path} (행: {len(df)}, 열: {len(df.columns)})")
        return df
    except Exception as e:
        logger.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        raise

def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
    """가격 시계열로부터 수익률을 계산합니다.
    
    Args:
        prices: 가격 시계열
        method: 'log' 또는 'simple' 중 하나
        
    Returns:
        pd.Series: 계산된 수익률 시계열
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError("method는 'log' 또는 'simple'이어야 합니다.")
    
    return returns

def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """자산 가치 곡선으로부터 낙폭(Drawdown)을 계산합니다.
    
    Args:
        equity_curve: 자산 가치 시계열
        
    Returns:
        pd.Series: 낙폭 시계열 (0~1 사이의 값)
    """
    peak = equity_curve.expanding(min_periods=1).max()
    return (equity_curve - peak) / peak

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """샤프 지수를 계산합니다.
    
    Args:
        returns: 일간 수익률 시계열
        risk_free_rate: 무위험 수익률 (연율)
        periods_per_year: 연간 거래일 수
        
    Returns:
        float: 연간화된 샤프 지수
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * excess_returns.mean() / (excess_returns.std() + 1e-10)

def calculate_calmar_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """칼마 지수를 계산합니다.
    
    Args:
        equity_curve: 자산 가치 시계열
        risk_free_rate: 무위험 수익률 (연율)
        periods_per_year: 연간 거래일 수
        
    Returns:
        float: 칼마 지수
    """
    returns = equity_curve.pct_change().dropna()
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (periods_per_year / len(returns)) - 1
    max_dd = calculate_drawdown(equity_curve).min()
    
    if max_dd == 0:
        return float('inf')
    
    return (cagr - risk_free_rate) / abs(max_dd)

def format_currency(value: float, currency: str = 'USD') -> str:
    """통화 형식으로 값을 포맷팅합니다.
    
    Args:
        value: 포맷팅할 값
        currency: 통화 코드 (USD, KRW 등)
        
    Returns:
        str: 포맷팅된 문자열
    """
    if currency.upper() == 'USD':
        return f"${value:,.2f}"
    elif currency.upper() == 'KRW':
        return f"₩{value:,.0f}"
    else:
        return f"{value:,.2f} {currency}"

def format_percent(value: float, decimals: int = 2) -> str:
    """백분율 형식으로 값을 포맷팅합니다.
    
    Args:
        value: 포맷팅할 값 (0~1 범위)
        decimals: 소수점 이하 자릿수
        
    Returns:
        str: 포맷팅된 문자열 (예: "12.34%")
    """
    return f"{value:.{decimals}%}"

def get_project_root() -> Path:
    """프로젝트 루트 디렉토리 경로를 반환합니다."""
    return Path(__file__).parent.parent.parent

class Timer:
    """코드 실행 시간을 측정하는 컨텍스트 매니저"""
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        self.elapsed = self.end_time - self.start_time
        logger.info(f"실행 시간: {self.elapsed.total_seconds():.2f}초")
    
    def get_elapsed_time(self) -> float:
        """경과 시간을 초 단위로 반환합니다."""
        return self.elapsed.total_seconds()
