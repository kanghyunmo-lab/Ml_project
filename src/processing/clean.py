"""
데이터 정제(Data Cleaning) 모듈

이 모듈은 데이터 품질 감사 결과를 바탕으로 원본 데이터를 정제하고,
분석 및 모델링에 적합한 형태로 변환하는 기능을 제공합니다.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

# 로깅 설정
logger.add(
    "../logs/data_cleaning.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

def clean_ohlcv_data(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    OHLCV 데이터를 정제하는 함수
    
    Args:
        df (pd.DataFrame): 정제할 OHLCV 데이터프레임
        save_path (Optional[Union[str, Path]], optional): 정제된 데이터를 저장할 경로. 기본값은 None.
        
    Returns:
        pd.DataFrame: 정제된 데이터프레임
    """
    try:
        logger.info("OHLCV 데이터 정제를 시작합니다...")
        
        # DataCleaner 인스턴스 생성 (기본 설정 사용)
        cleaner = DataCleaner()
        
        # 데이터 정제 실행
        cleaned_df = cleaner.clean_data(df)
        
        # 결과 저장 경로가 제공된 경우 저장
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일 확장자에 따라 저장 포맷 결정
            if save_path.suffix == '.parquet':
                cleaned_df.to_parquet(save_path, index=False)
            elif save_path.suffix == '.csv':
                cleaned_df.to_csv(save_path, index=False)
            else:
                # 기본값으로 parquet 형식 사용
                save_path = save_path.with_suffix('.parquet')
                cleaned_df.to_parquet(save_path, index=False)
            
            logger.info(f"정제된 데이터가 저장되었습니다: {save_path}")
        
        logger.info("OHLCV 데이터 정제가 완료되었습니다.")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"OHLCV 데이터 정제 중 오류가 발생했습니다: {str(e)}")
        raise

class DataCleanerConfig(BaseModel):
    """데이터 정제를 위한 설정 클래스"""
    
    # 결측치 처리 전략
    missing_value_strategy: Dict[str, str] = Field(
        default_factory=lambda: {
            'price': 'ffill',  # 가격 데이터는 이전 값으로 채우기
            'volume': 'fill_zero',  # 거래량은 0으로 채우기
            'default': 'drop'  # 그 외는 행 삭제
        },
        description="결측치 처리 전략 (ffill, bfill, fill_zero, drop, mean, median, mode)"
    )
    
    # 이상치 처리
    outlier_detection: Dict[str, float] = Field(
        default_factory=lambda: {
            'z_score_threshold': 3.0,  # Z-점수 기준
            'iqr_multiplier': 1.5  # IQR 승수
        },
        description="이상치 탐지 파라미터"
    )
    
    # 데이터 타입 변환
    dtypes: Dict[str, str] = Field(
        default_factory=lambda: {
            'open': 'float32',
            'high': 'float32',
            'low': 'float32',
            'close': 'float32',
            'volume': 'float32',
            'datetime': 'datetime64[ns]'
        },
        description="컬럼별 데이터 타입 지정"
    )

class DataCleaner:
    """데이터 정제를 수행하는 클래스"""
    
    def __init__(self, config: Optional[DataCleanerConfig] = None):
        """
        DataCleaner 초기화
        
        Args:
            config (DataCleanerConfig, optional): 데이터 정제 설정. 기본값은 None
        """
        self.config = config or DataCleanerConfig()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """설정 유효성 검증"""
        valid_strategies = ['ffill', 'bfill', 'fill_zero', 'drop', 'mean', 'median', 'mode']
        
        for col, strategy in self.config.missing_value_strategy.items():
            if strategy not in valid_strategies:
                raise ValueError(
                    f"잘못된 결측치 처리 전략: {strategy}. "
                    f"사용 가능한 전략: {', '.join(valid_strategies)}"
                )
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        if df.isnull().sum().sum() == 0:
            logger.info("결측치가 없습니다.")
            return df
            
        df_cleaned = df.copy()
        
        # 컬럼별로 처리 전략 적용
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() == 0:
                continue
                
            # 컬럼 유형에 따라 전략 선택
            strategy = (
                self.config.missing_value_strategy.get(col) or
                self.config.missing_value_strategy.get('price' if 'price' in col.lower() else 'default') or
                self.config.missing_value_strategy['default']
            )
            
            logger.info(f"컬럼 '{col}'의 결측치 처리: {strategy}")
            
            if strategy == 'ffill':
                df_cleaned[col] = df_cleaned[col].ffill()
            elif strategy == 'bfill':
                df_cleaned[col] = df_cleaned[col].bfill()
            elif strategy == 'fill_zero':
                df_cleaned[col] = df_cleaned[col].fillna(0)
            elif strategy == 'mean':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif strategy == 'median':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif strategy == 'mode':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            elif strategy == 'drop':
                df_cleaned = df_cleaned.dropna(subset=[col])
        
        return df_cleaned
    
    def detect_outliers(self, df: pd.DataFrame, column: str) -> pd.Series:
        """이상치 탐지 (Z-score 기반)"""
        if column not in df.columns:
            raise ValueError(f"컬럼 '{column}'을(를) 찾을 수 없습니다.")
            
        # Z-score 계산 (더 엄격한 임계값 사용)
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > 2.0  # 임계값을 2.0으로 낮춤
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """이상치 처리"""
        df_cleaned = df.copy()
        
        # 가격 및 거래량 컬럼에 대해 이상치 처리
        price_cols = [col for col in df.columns if col in ['open', 'high', 'low', 'close', 'volume']]
        
        for col in price_cols:
            if col not in df_cleaned.columns:
                continue
                
            # Z-score 기반 이상치 탐지
            is_outlier = self.detect_outliers(df_cleaned, col)
            outlier_count = is_outlier.sum()
            
            if outlier_count > 0:
                logger.info(f"컬럼 '{col}'에서 {outlier_count}개의 이상치 발견")
                
                if method == 'clip':
                    # IQR 기반으로 이상치 클리핑
                    q1 = df_cleaned[col].quantile(0.25)
                    q3 = df_cleaned[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr  # IQR 승수를 1.5로 고정
                    upper_bound = q3 + 1.5 * iqr
                    
                    # 클리핑 적용
                    df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
                    logger.info(f"컬럼 '{col}'의 이상치를 {lower_bound:.2f} ~ {upper_bound:.2f} 범위로 클리핑")
                    
                elif method == 'remove':
                    # 이상치가 있는 행 제거
                    df_cleaned = df_cleaned[~is_outlier]
                    
        return df_cleaned
    
    def convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 변환"""
        df_converted = df.copy()
        
        for col, dtype in self.config.dtypes.items():
            if col in df_converted.columns:
                try:
                    df_converted[col] = df_converted[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"컬럼 '{col}'을(를) {dtype}으로 변환하는 중 오류 발생: {e}")
        
        return df_converted
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 정제 파이프라인 실행"""
        logger.info("데이터 정제를 시작합니다...")
        
        # 1. 데이터 타입 변환
        df_cleaned = self.convert_dtypes(df)
        
        # 2. 결측치 처리
        df_cleaned = self.handle_missing_values(df_cleaned)
        
        # 3. 이상치 처리
        df_cleaned = self.handle_outliers(df_cleaned, method='clip')
        
        logger.info("데이터 정제가 완료되었습니다.")
        return df_cleaned


def save_cleaned_data(
    df: pd.DataFrame, 
    symbol: str, 
    timeframe: str,
    output_dir: str = "../data/processed"
) -> str:
    """정제된 데이터를 저장합니다."""
    from pathlib import Path
    import os
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 파일명 생성 (예: btc_usdt_4h_20200101_20230627_cleaned.parquet)
    start_date = df['datetime'].min().strftime("%Y%m%d")
    end_date = df['datetime'].max().strftime("%Y%m%d")
    filename = f"{symbol.lower().replace('/', '_')}_{timeframe}_{start_date}_{end_date}_cleaned.parquet"
    filepath = output_path / filename
    
    # Parquet 형식으로 저장
    df.to_parquet(filepath, index=False)
    
    logger.info(f"정제된 데이터가 저장되었습니다: {filepath}")
    return str(filepath)


def load_clean_and_save(
    input_path: str, 
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    config: Optional[DataCleanerConfig] = None,
    output_dir: str = "../data/processed"
) -> Tuple[pd.DataFrame, str]:
    """데이터를 로드하고 정제한 후 저장합니다."""
    from pathlib import Path
    
    # 데이터 로드
    logger.info(f"데이터 로드 중: {input_path}")
    df = pd.read_parquet(input_path)
    
    # 정제 실행
    cleaner = DataCleaner(config)
    df_cleaned = cleaner.clean_data(df)
    
    # 저장
    output_path = save_cleaned_data(df_cleaned, symbol, timeframe, output_dir)
    
    return df_cleaned, output_path


def main():
    """명령줄 인터페이스 진입점"""
    import argparse
    
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description='암호화폐 시장 데이터 정제 도구')
    parser.add_argument('input', help='입력 Parquet 파일 경로')
    parser.add_argument('--symbol', default='BTC/USDT', help='심볼 (예: BTC/USDT)')
    parser.add_argument('--timeframe', default='4h', help='타임프레임 (예: 1h, 4h, 1d)')
    parser.add_argument('--output-dir', default='../data/processed', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    try:
        # 기본 설정
        config = DataCleanerConfig(
            missing_value_strategy={
                'open': 'ffill',
                'high': 'ffill',
                'low': 'ffill',
                'close': 'ffill',
                'volume': 'fill_zero',
                'default': 'ffill'
            }
        )
        
        # 데이터 정제 실행
        df_cleaned, output_path = load_clean_and_save(
            input_path=args.input,
            symbol=args.symbol,
            timeframe=args.timeframe,
            config=config,
            output_dir=args.output_dir
        )
        
        print(f"\n=== 데이터 정제 완료 ===")
        print(f"입력 파일: {args.input}")
        print(f"출력 파일: {output_path}")
        print(f"처리된 행 수: {len(df_cleaned):,}")
        
        return 0
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        return 1

def clean_ohlcv_data(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    OHLCV 데이터를 정제하는 함수
    
    Args:
        df (pd.DataFrame): 정제할 OHLCV 데이터프레임
        save_path (Optional[Union[str, Path]], optional): 정제된 데이터를 저장할 경로. 기본값은 None.
        
    Returns:
        pd.DataFrame: 정제된 데이터프레임
    """
    try:
        logger.info("OHLCV 데이터 정제를 시작합니다...")
        
        # DataCleaner 인스턴스 생성 (기본 설정 사용)
        cleaner = DataCleaner()
        
        # 데이터 정제 실행
        cleaned_df = cleaner.clean_data(df)
        
        # 결과 저장 경로가 제공된 경우 저장
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일 확장자에 따라 저장 포맷 결정
            if save_path.suffix == '.parquet':
                cleaned_df.to_parquet(save_path, index=False)
            elif save_path.suffix == '.csv':
                cleaned_df.to_csv(save_path, index=False)
            else:
                # 기본값으로 parquet 형식 사용
                save_path = save_path.with_suffix('.parquet')
                cleaned_df.to_parquet(save_path, index=False)
            
            logger.info(f"정제된 데이터가 저장되었습니다: {save_path}")
        
        logger.info("OHLCV 데이터 정제가 완료되었습니다.")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"OHLCV 데이터 정제 중 오류가 발생했습니다: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    sys.exit(main())
