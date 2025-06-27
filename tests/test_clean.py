"""
데이터 정제 모듈 테스트

이 테스트는 src.processing.clean 모듈의 기능을 검증합니다.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# 테스트를 위해 루트 디렉토리를 모듈 경로에 추가
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.clean import DataCleaner, DataCleanerConfig, save_cleaned_data, load_clean_and_save

# 테스트용 데이터 생성
def create_test_data():
    """테스트용 데이터프레임 생성"""
    dates = pd.date_range('2023-01-01', periods=10, freq='4H')
    data = {
        'open': [100, 101, 102, np.nan, 104, 105, 106, 107, 108, 1000],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 1100],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 95],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 1000],
        'volume': [1000, 2000, np.nan, 4000, 5000, 6000, 7000, 8000, 9000, 100000],
        'datetime': dates
    }
    return pd.DataFrame(data)

class TestDataCleanerConfig:
    """DataCleanerConfig 클래스 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = DataCleanerConfig()
        
        # 기본값 검증
        assert config.missing_value_strategy['price'] == 'ffill'
        assert config.missing_value_strategy['volume'] == 'fill_zero'
        assert config.missing_value_strategy['default'] == 'drop'
        assert config.outlier_detection['z_score_threshold'] == 3.0
        assert config.outlier_detection['iqr_multiplier'] == 1.5


class TestDataCleaner:
    """DataCleaner 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 실행"""
        self.df = create_test_data()
        self.cleaner = DataCleaner()
    
    def test_handle_missing_values(self):
        """결측치 처리 테스트"""
        # ffill 전략으로 설정
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
        cleaner = DataCleaner(config)
        
        # 테스트 전 결측치 확인
        assert self.df.isnull().sum().sum() > 0
        
        # 결측치 처리
        cleaned_df = cleaner.handle_missing_values(self.df)
        
        # 결측치가 모두 처리되었는지 확인
        assert cleaned_df.isnull().sum().sum() == 0
        
        # volume 컬럼의 결측치가 0으로 채워졌는지 확인
        assert cleaned_df['volume'].iloc[2] == 0
        
        # open 컬럼의 결측치가 이전 값으로 채워졌는지 확인 (ffill 적용)
        # drop 대신 ffill을 사용하므로 행이 삭제되지 않음
        assert cleaned_df['open'].iloc[3] == 102  # 이전 값인 102로 채워짐
    
    def test_detect_outliers(self):
        """이상치 탐지 테스트"""
        # 마지막 행이 이상치인지 확인
        is_outlier = self.cleaner.detect_outliers(self.df, 'open')
        assert is_outlier.iloc[-1] == True  # 1000은 이상치
        assert is_outlier.iloc[0] == False   # 100은 정상
    
    def test_handle_outliers(self):
        """이상치 처리 테스트"""
        # 이상치 처리 (클리핑)
        cleaned_df = self.cleaner.handle_outliers(self.df, method='clip')
        
        # 이상치가 클리핑되었는지 확인
        assert cleaned_df['open'].iloc[-1] < 1000
        
        # IQR 범위 내에 있는지 확인
        q1 = self.df['open'].quantile(0.25)
        q3 = self.df['open'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        assert cleaned_df['open'].iloc[-1] <= upper_bound
    
    def test_convert_dtypes(self):
        """데이터 타입 변환 테스트"""
        # 타입 변환 실행
        converted_df = self.cleaner.convert_dtypes(self.df)
        
        # 타입 검증
        assert converted_df['open'].dtype == 'float32'
        assert converted_df['datetime'].dtype == 'datetime64[ns]'
    
    def test_clean_data_pipeline(self):
        """전체 정제 파이프라인 테스트"""
        # 전체 파이프라인 실행
        cleaned_df = self.cleaner.clean_data(self.df)
        
        # 결과 검증
        assert not cleaned_df.isnull().any().any()  # 결측치 없음
        assert cleaned_df['open'].dtype == 'float32'  # 타입 변환 확인
        
        # 이상치가 처리되었는지 확인 (마지막 행의 open 값이 클리핑됨)
        assert cleaned_df['open'].iloc[-1] < 1000


class TestSaveAndLoad:
    """데이터 저장 및 로드 테스트"""
    
    def test_save_cleaned_data(self, tmp_path):
        """정제된 데이터 저장 테스트"""
        df = create_test_data()
        output_path = save_cleaned_data(
            df, 
            'BTC/USDT', 
            '4h',
            output_dir=str(tmp_path)
        )
        
        # 파일이 생성되었는지 확인
        assert Path(output_path).exists()
        
        # 파일을 다시 로드하여 내용 확인
        loaded_df = pd.read_parquet(output_path)
        assert len(loaded_df) == len(df)
        assert 'datetime' in loaded_df.columns
    
    def test_load_clean_and_save(self, tmp_path):
        """전체 로드-정제-저장 파이프라인 테스트"""
        # 테스트용 임시 파일 생성
        test_file = tmp_path / "test_data.parquet"
        df = create_test_data()
        df.to_parquet(test_file)
        
        # 파이프라인 실행
        cleaned_df, output_path = load_clean_and_save(
            str(test_file),
            'BTC/USDT',
            '4h',
            output_dir=str(tmp_path / "processed")
        )
        
        # 결과 검증
        assert Path(output_path).exists()
        assert len(cleaned_df) > 0
        assert not cleaned_df.isnull().any().any()


class TestCommandLineInterface:
    """명령줄 인터페이스 테스트"""
    
    def test_cli_with_valid_file(self, tmp_path, monkeypatch, capsys):
        """유효한 파일로 CLI 테스트"""
        # 테스트용 임시 파일 생성
        test_file = tmp_path / "test_cli.parquet"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        df = create_test_data()
        df.to_parquet(test_file)

        # 명령줄 인수 모의
        monkeypatch.setattr(
            'sys.argv',
            ['test_clean.py', str(test_file), '--symbol', 'BTC/USDT', '--timeframe', '4h', '--output-dir', str(output_dir)]
        )

        # 모듈 직접 실행
        import src.processing.clean as clean_module
        
        # 모듈의 __name__이 "__main__"일 때만 실행되도록 수정
        if hasattr(clean_module, '__file__') and os.path.basename(clean_module.__file__) == 'clean.py':
            clean_module.main()

        # 출력 캡처
        captured = capsys.readouterr()
        print(captured.out)  # 디버깅을 위해 출력
        
        # 출력 디렉토리에 파일이 생성되었는지 확인
        output_files = list(output_dir.glob("*"))
        assert len(output_files) > 0, "출력 파일이 생성되지 않았습니다."
        
        # 정상 종료 메시지 확인
        assert "데이터 정제 완료" in captured.out


if __name__ == "__main__":
    pytest.main(["-v", __file__])
