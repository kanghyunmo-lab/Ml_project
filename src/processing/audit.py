"""
데이터 품질 감사 모듈

이 모듈은 수집된 암호화폐 시장 데이터의 품질을 검증하고, 
결측치, 이상치, 중복 데이터 등을 탐지하여 보고서를 생성합니다.
"""
import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
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
            float: lambda v: float(v) if not pd.isnull(v) else None
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
            mean = self.data[col].mean()
            std = self.data[col].std()
            if std > 0:  # 표준편차가 0이 아닌 경우에만 계산
                z_scores = ((self.data[col] - mean) / std).abs()
                outlier_mask = z_scores > z_threshold
                outlier_count = outlier_mask.sum()
                outlier_pct = (outlier_count / len(self.data)) * 100
                
                outlier_info[col] = {
                    'outlier_count': int(outlier_count),
                    'outlier_pct': round(float(outlier_pct), 2),
                    'max_z_score': round(float(z_scores.max()), 2)
                }
            else:
                outlier_info[col] = {
                    'outlier_count': 0,
                    'outlier_pct': 0.0,
                    'max_z_score': 0.0
                }
            
        return outlier_info
    
    def calculate_price_stats(self) -> Dict[str, Dict[str, float]]:
        """가격 관련 통계를 계산합니다."""
        stats = {}
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in self.data.columns:
                continue
                
            col_data = self.data[col].dropna()  # 결측치 제거
            if len(col_data) > 0:  # 데이터가 있는 경우에만 계산
                stats[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std() if len(col_data) > 1 else 0),
                    'min': float(col_data.min()),
                    '25%': float(col_data.quantile(0.25)),
                    '50%': float(col_data.median()),
                    '75%': float(col_data.quantile(0.75)),
                    'max': float(col_data.max()),
                    'count': int(col_data.count())
                }
            else:
                stats[col] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    '25%': 0.0,
                    '50%': 0.0,
                    '75%': 0.0,
                    'max': 0.0,
                    'count': 0
                }
            
        return stats
    
    def generate_report(self, z_threshold: float = 3.0) -> Dict[str, Any]:
        """
        데이터 품질 보고서를 생성합니다.
        
        Args:
            z_threshold (float): 이상치 탐지를 위한 Z-score 임계값
            
        Returns:
            dict: 데이터 품질 메트릭을 포함한 딕셔너리
        """
        logger.info(f"{self.symbol} 데이터 품질 감사를 시작합니다...")
        
        try:
            # 기본 메트릭 수집
            missing_values = self.check_missing_values()
            duplicate_rows = self.check_duplicates()
            gaps = self.check_timeline_integrity()
            
            # 통계 및 이상치 정보 수집
            price_stats = self.calculate_price_stats()
            outlier_info = self.detect_outliers(z_threshold)
            
            # 결과 딕셔너리 생성
            report = {
                'symbol': self.symbol,
                'start_date': str(self.data.index.min()),
                'end_date': str(self.data.index.max()),
                'total_rows': int(len(self.data)),
                'missing_values': missing_values,
                'duplicate_rows': int(duplicate_rows) if isinstance(duplicate_rows, (int, float)) else 0,
                'price_stats': price_stats,
                'outlier_info': outlier_info,
                'gaps_in_timeline': gaps or []
            }
            
            logger.info(f"{self.symbol} 데이터 품질 감사가 완료되었습니다.")
            return report
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {str(e)}")
            raise

def save_audit_report(report: Dict[str, Any], output_dir: str = "../reports") -> str:
    """
    감사 보고서를 JSON 파일로 저장합니다.
    
    Args:
        report (dict): 저장할 보고서 딕셔너리
        output_dir (str): 출력 디렉토리 경로
        
    Returns:
        str: 저장된 파일 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성 (현재 시간 기반)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"audit_report_{timestamp}.json")
    
    # JSON으로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"감사 보고서가 저장되었습니다: {output_path}")
    return output_path

def load_and_audit_data(filepath: str, symbol: str = "BTC/USDT") -> Dict[str, Any]:
    """
    Parquet 파일을 로드하고 감사를 수행합니다.
    
    Args:
        filepath (str): Parquet 파일 경로
        symbol (str): 심볼 (예: 'BTC/USDT')
        
    Returns:
        dict: 데이터 품질 메트릭을 포함한 딕셔너리
    """
    logger.info(f"데이터 로드 중: {filepath}")
    
    try:
        # 데이터 로드
        df = pd.read_parquet(filepath)
        logger.info(f"데이터 로드 완료: {len(df)}개의 행")
        
        # 감사 실행
        auditor = DataQualityAuditor(df, symbol)
        report = auditor.generate_report()
        
        if not report:
            raise ValueError("보고서 생성에 실패했습니다.")
        
        # 보고서 저장
        report_path = save_audit_report(report)
        logger.info(f"감사 보고서: {report_path}")
        
        # 결과 출력
        print("\n=== 데이터 품질 감사 결과 ===")
        print(f"심볼: {report.get('symbol', 'N/A')}")
        print(f"기간: {report.get('start_date', 'N/A')} ~ {report.get('end_date', 'N/A')}")
        print(f"총 행 수: {report.get('total_rows', 0):,}")
        print(f"중복 행 수: {report.get('duplicate_rows', 0)}")
        
        missing_values = report.get('missing_values', {})
        if isinstance(missing_values, dict):
            missing_count = len([k for k, v in missing_values.items() if v > 0])
            print(f"결측치가 있는 컬럼 수: {missing_count}")
        
        gaps = report.get('gaps_in_timeline', [])
        print(f"타임라인 누락 구간: {len(gaps)}개")
        
        return report
        
    except Exception as e:
        error_msg = f"오류가 발생했습니다: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        return {}
        
def run_data_audit(data: pd.DataFrame, symbol: str = "BTC/USDT") -> Dict[str, Any]:
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
