"""
모델 훈련 스크립트

이 스크립트는 피처와 타겟이 포함된 데이터를 사용하여 LightGBM 모델을 훈련합니다.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import joblib
import json
from datetime import datetime

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modeling.trainer import ModelTrainer

def load_data():
    """데이터를 로드하고 훈련/테스트 세트로 분할합니다."""
    # 데이터 로드
    data_path = Path('data/btc_usdt_4h_with_target.parquet')
    print(f"데이터 로드 중: {data_path}")
    data = pd.read_parquet(data_path)
    
    # 피처와 타겟 분리
    X = data.drop(columns=['target', 'timestamp', 'datetime'], errors='ignore')
    y = data['target']
    
    # 훈련/테스트 세트 분할 (시간순으로 80:20 분할)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"훈련 세트: {len(X_train)}개, 테스트 세트: {len(X_test)}개")
    print("훈련 세트 클래스 분포:")
    print(y_train.value_counts(normalize=True).sort_index())
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """모델을 훈련합니다."""
    # 클래스 가중치 계산 (불균형 데이터 처리)
    class_weights = {}
    class_counts = y_train.value_counts()
    total = len(y_train)
    
    for label in sorted(y_train.unique()):
        class_weights[int(label)] = (1 / class_counts[label]) * (total / len(class_counts))
    
    print("\n클래스 가중치:", class_weights)
    
    # 모델 훈련
    trainer = ModelTrainer()
    
    # 모델 파라미터 (필요에 따라 조정 가능)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'class_weight': class_weights  # 클래스 가중치 적용
    }
    
    print("\n모델 훈련을 시작합니다...")
    metrics = trainer.train(
        X_train, y_train,
        n_splits=5,  # 시계열 교차검증 폴드 수
        n_trials=50,  # Optuna 최적화 시도 횟수
        optimize=True,  # 하이퍼파라미터 최적화 활성화
        early_stopping_rounds=50
    )
    
    return trainer, metrics

def evaluate_model(trainer, X_test, y_test):
    """모델을 평가합니다."""
    print("\n테스트 세트에서 모델 평가 중...")
    
    # 예측
    y_pred = trainer.predict(X_test)
    
    # 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n테스트 세트 성능:")
    print(f"정확도: {accuracy:.4f}")
    print(f"가중 F1 점수: {f1:.4f}")
    
    # 분류 보고서
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred, target_names=['하락(-1)', '횡보(0)', '상승(1)']))
    
    # 혼동 행렬
    print("\n혼동 행렬:")
    print(confusion_matrix(y_test, y_pred))
    
    # 특성 중요도
    if hasattr(trainer, 'feature_importances_'):
        print("\n상위 10개 특성 중요도:")
        importances = trainer.feature_importances_
        feature_importances = pd.DataFrame({
            'feature': X_test.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        print(feature_importances.head(10))

def save_model(trainer):
    """모델을 저장합니다."""
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # 모델 저장
    model_path = model_dir / 'lightgbm_model'
    trainer.save_model(str(model_path))
    
    # 모델 메타데이터 저장
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'model_type': 'LightGBM',
        'classes_': trainer.model.classes_.tolist() if hasattr(trainer.model, 'classes_') else None,
        'feature_names': trainer.model.feature_name_ if hasattr(trainer.model, 'feature_name_') else None
    }
    
    with open(model_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n모델이 저장되었습니다: {model_path}")

def main():
    """메인 함수"""
    try:
        # 데이터 로드
        X_train, X_test, y_train, y_test = load_data()
        
        # 모델 훈련
        trainer, metrics = train_model(X_train, y_train)
        
        # 모델 평가
        evaluate_model(trainer, X_test, y_test)
        
        # 모델 저장
        save_model(trainer)
        
        print("\n모델 훈련 및 평가가 완료되었습니다!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
