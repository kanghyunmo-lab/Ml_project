"""
데이터 품질 감사 모듈

이 모듈은 수집된 암호화폐 시장 데이터의 품질을 검증하고, 
결측치, 이상치, 중복 데이터 등을 탐지하여 보고서를 생성합니다.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger
from pydantic import BaseModel, Field, validator

# 로깅 설정
logger.add(
    "../logs/data_audit.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

class DataQualityMetrics(BaseModel):
    """데이터 품질 지표를 저장하는 Pydantic 모델"""
    
    # 기본 메트릭
    start_date: str
    end_date: str
    total_rows: int
    missing_values: Dict[str, int] = Field(..., description="컬럼별 결측치 수")
    duplicate_rows: int = Field(..., description="중복 행 수")
    
    # 통계적 메트릭
    price_stats: Dict[str, Dict[str, float]] = Field(
        ..., 
        description="가격 관련 컬럼의 통계 (open, high, low, close, volume)"
    )
    
    # 이상치 관련 메트릭
    outlier_info: Dict[str, Dict[str, float]] = Field(
        ...,
        description="이상치 관련 정보 (Z-score, IQR 기반)"
    )
    
    # 데이터 무결성 검사
    gaps_in_timeline: List[str] = Field(
        default_factory=list,
        description="타임라인 상의 누락된 기간 목록"
    )
    
    class Config:
        json_encoders = {
            np.float64: lambda v: float(v) if not np.isnan(v) else None
        }

class DataQualityAuditor:
    """데이터 품질을 감사하고 보고서를 생성하는 클래스"""
    
    def __init__(self, data: pd.DataFrame, symbol: str = "BTC/USDT"):
        """
        DataQualityAuditor 초기화
        
        Args:
            data (pd.DataFrame): 감사할 데이터 (반드시 'timestamp' 또는 'datetime' 컬럼 포함)
            symbol (str): 심볼 (예: 'BTC/USDT')
        """
        self.data = data.copy()
        self.symbol = symbol
        
        # 타임스탬프 컬럼 정규화
        self._normalize_timestamps()
        
        # 기본 컬럼 확인
        self.required_columns = {'open', 'high', 'low', 'close', 'volume'}
        self._validate_columns()
        
    def _normalize_timestamps(self) -> None:
        """타임스탬프 컬럼을 정규화합니다."""
        if 'datetime' not in self.data.columns and 'timestamp' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='ms')
        
        if 'datetime' not in self.data.columns:
            raise ValueError("데이터프레임에 'datetime' 또는 'timestamp' 컬럼이 필요합니다.")
            
        self.data = self.data.sort_values('datetime').reset_index(drop=True)
    
    def _validate_columns(self) -> None:
        """필수 컬럼이 있는지 검증합니다."""
        missing_columns = self.required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
    
    def check_missing_values(self) -> Dict[str, int]:
        """결측치를 확인합니다."""
        return self.data.isnull().sum().to_dict()
    
    def check_duplicates(self) -> int:
        """중복 행을 확인합니다."""
        return self.data.duplicated().sum()
    
    def check_timeline_integrity(self, freq: str = '4H') -> List[str]:
        """타임라인의 연속성을 확인합니다."""
        # 예상되는 날짜 범위 생성
        expected_dates = pd.date_range(
            start=self.data['datetime'].min(),
            end=self.data['datetime'].max(),
            freq=freq
        )
        
        # 누락된 날짜 찾기
        missing_dates = set(expected_dates) - set(self.data['datetime'])
        return sorted([str(date) for date in missing_dates])
    
    def detect_outliers(self, z_threshold: float = 3.0) -> Dict[str, Dict[str, float]]:
        """이상치를 탐지합니다 (Z-score 기반)."""
        outlier_info = {}
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in self.data.columns:
                continue
                
            # Z-score 계산
            z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
            
            # 이상치 통계
            outlier_mask = z_scores > z_threshold
            outlier_count = outlier_mask.sum()
            outlier_pct = (outlier_count / len(self.data)) * 100
            
            outlier_info[col] = {
                'outlier_count': int(outlier_count),
                'outlier_pct': round(float(outlier_pct), 2),
                'max_z_score': round(float(z_scores.max()), 2)
            }
            
        return outlier_info
    
    def calculate_price_stats(self) -> Dict[str, Dict[str, float]]:
        """가격 관련 통계를 계산합니다."""
        stats = {}
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in self.data.columns:
                continue
                
            stats[col] = {
                'mean': float(self.data[col].mean()),
                'std': float(self.data[col].std()),
                'min': float(self.data[col].min()),
                '25%': float(self.data[col].quantile(0.25)),
                '50%': float(self.data[col].median()),
                '75%': float(self.data[col].quantile(0.75)),
                'max': float(self.data[col].max()),
                'skew': float(self.data[col].skew()),
                'kurtosis': float(self.data[col].kurtosis())
            }
            
        return stats
    
    def generate_report(self, z_threshold: float = 3.0) -> DataQualityMetrics:
        """데이터 품질 보고서를 생성합니다."""
        logger.info(f"{self.symbol} 데이터 품질 감사를 시작합니다...")
        
        # 기본 정보
        start_date = self.data['datetime'].min().strftime('%Y-%m-%d %H:%M:%S')
        end_date = self.data['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
        total_rows = len(self.data)
        
        # 데이터 품질 메트릭 수집
        missing_values = self.check_missing_values()
        duplicate_rows = self.check_duplicates()
        gaps = self.check_timeline_integrity()
        price_stats = self.calculate_price_stats()
        outlier_info = self.detect_outliers(z_threshold)
        
        # 보고서 생성
        report = DataQualityMetrics(
            start_date=start_date,
            end_date=end_date,
            total_rows=total_rows,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            price_stats=price_stats,
            outlier_info=outlier_info,
            gaps_in_timeline=gaps
        )
        
        logger.info(f"{self.symbol} 데이터 품질 감사가 완료되었습니다.")
        return report

def save_audit_report(report: DataQualityMetrics, output_dir: str = "../reports") -> str:
    """감사 보고서를 JSON 파일로 저장합니다."""
    import json
    from pathlib import Path
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 파일명 생성 (예: audit_report_BTC_USDT_20230627_123456.json)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    symbol = "_" + report.symbol.replace("/", "_") if hasattr(report, 'symbol') else ""
    filename = f"audit_report{symbol}_{timestamp}.json"
    filepath = output_path / filename
    
    # JSON으로 저장
    with open(filepath, 'w') as f:
        json.dump(report.dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"감사 보고서가 저장되었습니다: {filepath}")
    return str(filepath)

def load_and_audit_data(filepath: str, symbol: str = "BTC/USDT") -> DataQualityMetrics:
    """Parquet 파일을 로드하고 감사를 수행합니다."""
    logger.info(f"데이터 로드 중: {filepath}")
    
    # 데이터 로드
    try:
        df = pd.read_parquet(filepath)
        logger.info(f"데이터 로드 완료: {len(df)}개의 행")
        
        # 감사 수행
        auditor = DataQualityAuditor(df, symbol=symbol)
        report = auditor.generate_report()
        
        # 보고서 저장
        report_path = save_audit_report(report)
        logger.info(f"감사 보고서: {report_path}")
        
        return report.dict()
        
    except Exception as e:
        logger.error(f"데이터 로드 또는 감사 중 오류 발생: {e}")
        raise

def run_data_audit(data: pd.DataFrame, symbol: str = "BTC/USDT") -> 'DataQualityMetrics':
    """
    데이터 품질 감사를 실행하고 결과를 반환합니다.
    
    Args:
        data (pd.DataFrame): 감사할 데이터프레임
        symbol (str, optional): 심볼 (예: 'BTC/USDT'). 기본값은 "BTC/USDT".
        
    Returns:
        DataQualityMetrics: 데이터 품질 지표를 포함한 객체
    """
    try:
        # 데이터 품질 감사기 초기화
        auditor = DataQualityAuditor(data, symbol)
        
        # 감사 실행 및 보고서 생성
        report_obj = auditor.generate_report()
        report = report_obj.dict()
        
        # 감사 결과 로깅 (객체 사용)
        logger.info(f"데이터 품질 감사가 완료되었습니다. {report_obj.total_rows}개의 행을 분석했습니다.")
        logger.info(f"- 결측치: {sum(report_obj.missing_values.values())}개")
        logger.info(f"- 중복 행: {report_obj.duplicate_rows}개")
        logger.info(f"- 누락된 기간: {len(report_obj.gaps_in_timeline)}개")
        
        return report
        
    except Exception as e:
        logger.error(f"데이터 품질 감사 중 오류가 발생했습니다: {str(e)}")
        raise

if __name__ == "__main__":
    # 사용 예시
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        symbol = sys.argv[2] if len(sys.argv) > 2 else "BTC/USDT"
        
        try:
            report = load_and_audit_data(filepath, symbol)
            print(f"\n감사 보고서가 성공적으로 생성되었습니다. {report.total_rows}개의 행을 분석했습니다.")
            print(f"- 결측치: {sum(report.missing_values.values())}개")
            print(f"- 중복 행: {report.duplicate_rows}개")
            print(f"- 누락된 기간: {len(report.gaps_in_timeline)}개")
            
        except Exception as e:
            print(f"오류가 발생했습니다: {str(e)}")
    else:
        print("사용법: python -m src.processing.audit <parquet_file_path> [symbol]")
        print("예시: python -m src.processing.audit ../data/raw/btc_usdt_4h.parquet BTC/USDT")
