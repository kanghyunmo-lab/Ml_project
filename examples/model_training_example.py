"""
ModelTrainer 사용 예제 스크립트

이 스크립트는 ModelTrainer 클래스를 사용하여 모델을 훈련하고 평가하는 방법을 보여줍니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 루트 디렉토리 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.modeling.trainer import ModelTrainer
from src.processing.feature_engineer import FeatureEngineer
from src.modeling.labeler import TripleBarrierLabeler, TripleBarrierParams

# 로깅 설정
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data():
    """샘플 데이터 로드"""
    # 실제로는 데이터베이스나 파일에서 로드
    # 여기서는 임의의 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='4H')
    
    # 가격 데이터 생성 (랜덤 워크)
    log_returns = np.random.normal(0.0001, 0.01, size=len(dates))
    prices = 50000 * np.exp(np.cumsum(log_returns))
    
    # OHLCV 데이터 생성
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    # 간단한 매수 신호 생성 (랜덤)
    data['signal'] = np.random.choice([0, 1], size=len(data), p=[0.9, 0.1])
    
    return data

def main():
    """메인 실행 함수"""
    logger.info("Starting model training example...")
    
    # 1. 데이터 로드
    logger.info("Loading sample data...")
    data = load_sample_data()
    
    # 2. 피처 엔지니어링
    logger.info("Engineering features...")
    # FeatureEngineer 초기화 시 데이터 전달
    feature_engineer = FeatureEngineer(data)
    features = feature_engineer.add_technical_indicators()
    
    # 3. 레이블 생성 (삼중 장벽 기법)
    logger.info("Generating labels using Triple Barrier Method...")
    params = TripleBarrierParams(
        take_profit=1.5,
        stop_loss=1.0,
        max_holding_period=24,  # 24 * 4h = 96h (4일)
        volatility_window=20,
        volatility_scale=2.0
    )
    labeler = TripleBarrierLabeler(params)
    labels = labeler.generate_labels(features, signal_col='signal', price_col='close')
    
    # 레이블 분포 확인
    label_dist = labels.value_counts(normalize=True)
    logger.info(f"Label distribution:\n{label_dist}")
    
    # 4. 훈련 데이터 준비
    X = features.drop(columns=['open', 'high', 'low', 'close', 'volume', 'signal'], errors='ignore')
    y = labels
    
    # NaN이 있는 행 제거
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    logger.info(f"Final dataset shape: {X.shape}, Labels: {y.shape}")
    
    # 5. 모델 훈련
    logger.info("Training model...")
    trainer = ModelTrainer()
    
    # 하이퍼파라미터 최적화 및 훈련
    metrics = trainer.train(
        X, y,
        n_splits=3,           # 교차검증 폴드 수
        optimize=True,        # 하이퍼파라미터 최적화 수행
        n_trials=20,          # 최적화 시도 횟수
        early_stopping_rounds=50
    )
    
    # 6. 결과 출력
    logger.info("\nModel Training Results:")
    logger.info(f"Average Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Average F1 Score: {metrics['f1_weighted']:.4f}")
    logger.info("\nConfusion Matrix (Sum of all folds):")
    logger.info(metrics['confusion_matrix'])
    
    # 7. 특성 중요도 시각화
    if hasattr(trainer, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        trainer.feature_importances_.head(15).plot(kind='barh')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        
        # 디렉토리 생성
        os.makedirs('results', exist_ok=True)
        
        # 이미지 저장
        plt.savefig('results/feature_importances.png')
        logger.info("Feature importance plot saved to 'results/feature_importances.png'")
    
    # 8. 모델 저장
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/lightgbm_model')
    logger.info("Model saved to 'models/lightgbm_model'")
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    main()
