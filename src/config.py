"""
프로젝트 설정 파일
"""
import os
from pathlib import Path

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(__file__).parent.parent

# 데이터 디렉토리
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 모델 디렉토리
MODELS_DIR = ROOT_DIR / "models"

# 결과 디렉토리
RESULTS_DIR = ROOT_DIR / "results"

# 로그 디렉토리
LOGS_DIR = ROOT_DIR / "logs"

# 테스트 디렉토리
TEST_DIR = ROOT_DIR / "tests"

# 바이낸스 API 설정 (실제로는 환경 변수에서 로드해야 함)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# 백테스팅 설정
INITIAL_CAPITAL = 10000.0  # 초기 자본 (USDT)
COMMISSION = 0.0004  # 0.04% 수수료
LEVERAGE = 3.0  # 기본 레버리지
RISK_PER_TRADE = 0.02  # 거래당 2% 리스크

# 모델 설정
MODEL_NAME = "lgbm_model.pkl"

# 필요한 디렉토리 생성 함수
def create_directories(directories):
    """필요한 디렉토리들을 생성합니다."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# 프로젝트 시작 시 필요한 디렉토리 생성
required_dirs = [
    DATA_DIR, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    MODELS_DIR, 
    RESULTS_DIR, 
    LOGS_DIR
]

create_directories(required_dirs)

# 로깅 설정
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOGS_DIR / 'quantum_leaf.log',
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8',
        },
    },
    'loggers': {
        '': {  # 루트 로거
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'src': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}
