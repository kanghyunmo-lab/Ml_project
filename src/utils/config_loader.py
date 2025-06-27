"""
설정 파일 로더 유틸리티

YAML 형식의 설정 파일을 로드하고 접근하기 위한 유틸리티 모듈입니다.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    """설정 파일 로더 클래스"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        설정 파일을 로드합니다.
        
        Args:
            config_path: 설정 파일 경로. None인 경우 기본 경로에서 시도합니다.
            
        Returns:
            로드된 설정 딕셔너리
        """
        if not cls._config:
            # 기본 설정 파일 경로
            if config_path is None:
                # 1. 현재 작업 디렉토리의 config 디렉토리 확인
                config_path = Path('config/data_config.yaml')
                
                # 2. 프로젝트 루트의 config 디렉토리 확인 (src/utils/에서 실행되는 경우)
                if not config_path.exists():
                    config_path = Path(__file__).parent.parent.parent / 'config' / 'data_config.yaml'
            else:
                config_path = Path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(
                    f"설정 파일을 찾을 수 없습니다. 다음 위치에서 확인해주세요:\n"
                    f"1. {Path('config/data_config.yaml').absolute()}\n"
                    f"2. {Path(__file__).parent.parent.parent / 'config' / 'data_config.yaml'}"
                )
            
            print(f"설정 파일 로드 중: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f)
        
        return cls._config
    
    @classmethod
    def get(cls, *keys, default=None) -> Any:
        """
        설정 값을 가져옵니다.
        
        Args:
            *keys: 점(.)으로 구분된 키 경로 (예: 'data.files.btc_usdt_4h')
            default: 키가 없을 때 반환할 기본값
            
        Returns:
            설정 값 또는 기본값
        """
        if not cls._config:
            cls.load_config()
            
        result = cls._config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result

def get_data_path(file_key: str) -> Path:
    """
    데이터 파일의 전체 경로를 반환합니다.
    
    Args:
        file_key: data_config.yaml의 files 섹션에 정의된 파일 키
        
    Returns:
        Path: 데이터 파일의 전체 경로
    """
    config = ConfigLoader.load_config()
    
    # 기본 디렉토리 구성
    base_dir = Path(ConfigLoader.get('data', 'base_dir', default='data'))
    processed_dir = ConfigLoader.get('data', 'processed_dir', default='processed')
    
    # 파일명 가져오기
    filename = ConfigLoader.get('data', 'files', file_key)
    if not filename:
        raise KeyError(f"'{file_key}'에 해당하는 파일이 설정에 정의되어 있지 않습니다.")
    
    # 파일 경로 구성
    file_path = base_dir / processed_dir / filename
    
    # 파일 존재 여부 확인
    if not file_path.exists():
        # 상대 경로로도 시도 (Jupyter 노트북 등에서 실행 시)
        alt_path = Path('data') / processed_dir / filename
        if alt_path.exists():
            return alt_path
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다. 다음 위치에서 확인해주세요:\n"
            f"1. {file_path}\n"
            f"2. {alt_path}"
        )
    
    return file_path

def get_model_dir() -> Path:
    """모델 저장 디렉토리 경로를 반환합니다."""
    model_dir = Path(ConfigLoader.get('model', 'output_dir', default='models'))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
