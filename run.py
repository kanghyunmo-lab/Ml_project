"""
Quantum Leaf - 메인 실행 스크립트

이 스크립트는 데이터 수집부터 백테스팅까지 전체 파이프라인을 실행합니다.

사용법:
    python run.py [모드] [옵션]

모드:
    all: 전체 파이프라인 실행 (데이터 수집 → 전처리 → 피처 엔지니어링 → 모델 훈련 → 백테스트)
    train: 모델 훈련만 실행
    backtest: 백테스트만 실행
    predict: 예측만 실행 (실전 투입용)

옵션:
    --config: 설정 파일 경로 (기본값: config/config.yaml)
    --start-date: 데이터 시작일 (YYYY-MM-DD)
    --end-date: 데이터 종료일 (YYYY-MM-DD)
    --symbol: 거래 심볼 (기본값: BTC/USDT)
    --timeframe: 시간봉 (기본값: 4h)

예시:
    # 전체 파이프라인 실행
    python run.py all --start-date 2023-01-01 --end-date 2024-01-01
    
    # 모델 훈련만 실행
    python run.py train --config config/train_config.yaml
    
    # 백테스트만 실행
    python run.py backtest --model models/lgbm_model.pkl
"""

import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import logging

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

# 모듈 임포트
from src.collection.binance_collector import BinanceDataCollector
from src.processing.clean import DataCleaner
from src.processing.feature_engineer import FeatureEngineer
from src.modeling.labeler import TripleBarrierLabeler
from src.modeling.trainer import ModelTrainer
from src.backtesting.engine import BacktestEngine, TripleBarrierStrategy
from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger("quantum_leaf")

class QuantumLeafPipeline:
    """퀀텀 리프 전체 파이프라인을 관리하는 클래스"""
    
    def __init__(self, config_path="config/config.yaml"):
        """초기화"""
        self.config = self._load_config(config_path)
        self.data = None
        self.model = None
        self.results = None
    
    def _load_config(self, config_path):
        """설정 파일 로드"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"설정 파일을 로드하는 중 오류 발생: {e}")
            raise
    
    def collect_data(self, symbol="BTC/USDT", timeframe="4h", start_date=None, end_date=None):
        """데이터 수집"""
        logger.info("데이터 수집을 시작합니다...")
        
        # 기본값 설정
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=365)  # 기본 1년치 데이터
        
        # 데이터 수집기 초기화
        collector = BinanceDataCollector()
        
        # 데이터 수집
        self.data = collector.collect_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"데이터 수집 완료: {len(self.data)}개의 봉")
        return self.data
    
    def preprocess_data(self):
        """데이터 전처리"""
        if self.data is None:
            raise ValueError("먼저 데이터를 수집해주세요.")
            
        logger.info("데이터 전처리를 시작합니다...")
        cleaner = DataCleaner()
        self.data = cleaner.clean(self.data)
        logger.info("데이터 전처리 완료")
        return self.data
    
    def extract_features(self):
        """피처 엔지니어링"""
        if self.data is None:
            raise ValueError("먼저 데이터를 수집하고 전처리해주세요.")
            
        logger.info("피처 엔지니어링을 시작합니다...")
        feature_engineer = FeatureEngineer()
        features = feature_engineer.transform(self.data)
        logger.info(f"피처 엔지니어링 완료: {features.shape[1]}개의 피처 생성됨")
        return features
    
    def train_model(self, features, labels, save_path=None):
        """모델 훈련"""
        logger.info("모델 훈련을 시작합니다...")
        
        # 모델 초기화
        trainer = ModelTrainer(params=self.config.get('model_params', {}))
        
        # 모델 훈련
        metrics = trainer.train(
            features,
            labels,
            n_splits=self.config.get('n_splits', 5),
            optimize=self.config.get('optimize', True),
            n_trials=self.config.get('n_trials', 50)
        )
        
        # 모델 저장
        if save_path:
            trainer.save_model(save_path)
            logger.info(f"모델이 저장되었습니다: {save_path}")
        
        self.model = trainer
        logger.info("모델 훈련 완료")
        return metrics
    
    def run_backtest(self, data, model_path=None):
        """백테스트 실행"""
        logger.info("백테스트를 시작합니다...")
        
        # 백테스트 엔진 초기화
        engine = BacktestEngine(
            initial_capital=self.config.get('initial_capital', 10000.0),
            commission=self.config.get('commission', 0.0004),
            leverage=self.config.get('leverage', 3),
            slippage=self.config.get('slippage', 0.0005)
        )
        
        # 데이터 로드 (실제로는 모델 예측 결과 포함된 데이터 필요)
        # 여기서는 예시로 원본 데이터를 사용하지만, 실제로는 예측 결과가 포함된 데이터를 사용해야 함
        data.to_parquet("temp_backtest_data.parquet")
        engine.load_data("temp_backtest_data.parquet")
        
        # 전략 추가
        engine.add_strategy(TripleBarrierStrategy)
        
        # 백테스트 실행
        results = engine.run_backtest()
        
        # 결과 저장
        report_path = engine.generate_report('reports')
        logger.info(f"백테스트 보고서가 생성되었습니다: {report_path}")
        
        self.results = results
        return results

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='퀀텀 리프 트레이딩 시스템')
    
    # 모드 선택
    parser.add_argument('mode', type=str, choices=['all', 'train', 'backtest', 'predict'],
                       help='실행할 모드 선택')
    
    # 공통 옵션
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='설정 파일 경로')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                      help='거래 심볼')
    parser.add_argument('--timeframe', type=str, default='4h',
                      help='시간봉 (예: 1h, 4h, 1d)')
    parser.add_argument('--start-date', type=str,
                      help='데이터 시작일 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                      help='데이터 종료일 (YYYY-MM-DD)')
    
    # 모델 관련 옵션
    parser.add_argument('--model', type=str,
                      help='사용할 모델 파일 경로')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_args()
    
    try:
        # 파이프라인 초기화
        pipeline = QuantumLeafPipeline(config_path=args.config)
        
        # 날짜 파싱
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
        
        # 모드에 따라 실행
        if args.mode == 'all':
            # 전체 파이프라인 실행
            pipeline.collect_data(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=start_date,
                end_date=end_date
            )
            pipeline.preprocess_data()
            features = pipeline.extract_features()
            
            # 레이블 생성 (예시로 랜덤 레이블 사용, 실제로는 TripleBarrierLabeler 사용)
            import numpy as np
            labels = np.random.choice([-1, 0, 1], size=len(features))
            
            # 모델 훈련
            pipeline.train_model(features, labels, save_path='models/latest_model.pkl')
            
            # 백테스트 실행
            pipeline.run_backtest(pipeline.data)
            
        elif args.mode == 'train':
            # 모델 훈련만 실행
            # (실제 구현에서는 저장된 데이터나 API에서 데이터 로드)
            pass
            
        elif args.mode == 'backtest':
            # 백테스트만 실행
            if not args.model:
                raise ValueError("백테스트를 실행하려면 --model 매개변수로 모델 파일을 지정해야 합니다.")
            # (실제 구현에서는 저장된 데이터나 API에서 데이터 로드)
            pass
            
        elif args.mode == 'predict':
            # 예측만 실행 (실전 투입용)
            if not args.model:
                raise ValueError("예측을 실행하려면 --model 매개변수로 모델 파일을 지정해야 합니다.")
            # (실제 구현에서는 최신 데이터를 가져와 예측 수행)
            pass
            
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
