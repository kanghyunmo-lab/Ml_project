"""
모델 훈련 파이프라인 모듈

이 모듈은 LightGBM 모델의 훈련, 검증, 최적화를 담당합니다.
PRD 문서의 '03_feature_and_model.md'에 명시된 요구사항을 준수합니다.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    LightGBM 모델의 훈련, 검증, 최적화를 담당하는 클래스
    
    PRD 문서 '03_feature_and_model.md'의 요구사항을 구현합니다.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        초기화 메서드
        
        Args:
            params (Dict[str, Any], optional): LightGBM 모델 파라미터. 
                기본값은 PRD에 명시된 권장값을 사용합니다.
        """
        self.model = None
        self.best_params = None
        self.feature_importances_ = None
        self.scaler = None
        
        # PRD에 명시된 기본 파라미터
        self.params = {
            'objective': 'multiclass',
            'num_class': 3,  # 상승(1), 하락(-1), 횡보(0)
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1  # 모든 CPU 코어 사용
        }
        
        # 사용자 정의 파라미터가 있으면 업데이트
        if params:
            self.params.update(params)
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_splits: int = 5,
        optimize: bool = True,
        n_trials: int = 50,
        early_stopping_rounds: int = 50
    ) -> Dict[str, Any]:
        """
        모델 훈련 및 최적화를 수행합니다.
        
        PRD 요구사항:
        - TimeSeriesSplit을 사용한 시계열 교차검증
        - Optuna를 사용한 하이퍼파라미터 최적화
        
        Args:
            X (pd.DataFrame): 학습 피처
            y (pd.Series): 타겟 레이블 (1, 0, -1)
            n_splits (int): 시계열 교차검증 폴드 수
            optimize (bool): 하이퍼파라미터 최적화 수행 여부
            n_trials (int): Optuna 최적화 시도 횟수
            early_stopping_rounds (int): 조기 종료를 위한 라운드 수
            
        Returns:
            Dict[str, Any]: 훈련 결과 메트릭
        """
        logger.info("Starting model training...")
        
        # 1. 하이퍼파라미터 최적화
        if optimize:
            logger.info(f"Optimizing hyperparameters with {n_trials} trials...")
            self.best_params = self._optimize_hyperparameters(X, y, n_trials, n_splits)
            self.params.update(self.best_params)
        
        # 2. 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        
        logger.info(f"Starting {n_splits}-fold time series cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # LightGBM 데이터셋으로 변환
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # 모델 훈련
            self.model = lgb.train(
                params=self.params,
                train_set=train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                    lgb.log_evaluation(period=10)
                ]
            )
            
            # 검증 세트 예측 및 평가
            y_pred = self.predict(X_val)
            metrics = self._evaluate(y_val, y_pred)
            fold_metrics.append(metrics)
            
            logger.info(f"Fold {fold + 1} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}")
        
        # 3. 전체 데이터로 최종 모델 훈련
        logger.info("Training final model on full dataset...")
        full_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            params=self.params,
            train_set=full_data,
            callbacks=[lgb.log_evaluation(period=10)]
        )
        
        # 4. 특성 중요도 저장
        self.feature_importances_ = pd.Series(
            self.model.feature_importance(importance_type='gain'),
            index=X.columns
        ).sort_values(ascending=False)
        
        # 5. 평균 메트릭 계산
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'f1_weighted': np.mean([m['f1_weighted'] for m in fold_metrics]),
            'confusion_matrix': sum([m['confusion_matrix'] for m in fold_metrics])
        }
        
        logger.info(f"Training completed. Avg Accuracy: {avg_metrics['accuracy']:.4f}")
        return avg_metrics
    
    def _optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_trials: int,
        n_splits: int
    ) -> Dict[str, Any]:
        """
        Optuna를 사용하여 하이퍼파라미터를 최적화합니다.
        
        PRD 요구사항: Optuna를 사용한 베이지안 최적화
        """
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
            }
            
            # 기본 파라미터와 병합
            model_params = self.params.copy()
            model_params.update(params)
            
            # 교차검증으로 성능 평가
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = lgb.train(
                    params=model_params,
                    train_set=lgb.Dataset(X_train, label=y_train),
                    valid_sets=[lgb.Dataset(X_val, label=y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                    verbose_eval=False
                )
                
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                y_pred = np.argmax(y_pred, axis=1)  # 확률을 클래스로 변환
                score = f1_score(y_val, y_pred, average='weighted')
                scores.append(score)
            
            return np.mean(scores)
        
        # Optuna 최적화 실행
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial: {study.best_trial.value}")
        return study.best_params
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        훈련된 모델로 예측을 수행합니다.
        
        Args:
            X (pd.DataFrame): 예측할 데이터
            
        Returns:
            np.ndarray: 예측 클래스 (1, 0, -1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # 예측 (확률 반환)
        proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        # 확률을 클래스로 변환 (0, 1, 2) -> (-1, 0, 1)
        return np.argmax(proba, axis=1) - 1
    
    def _evaluate(
        self, 
        y_true: Union[pd.Series, np.ndarray], 
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        모델 성능을 평가합니다.
        
        PRD 요구사항: 정확도, F1-score, 혼동 행렬
        
        Args:
            y_true: 실제 타겟 값 (-1, 0, 1)
            y_pred: 예측된 타겟 값 (-1, 0, 1)
            
        Returns:
            Dict[str, Any]: 평가 메트릭 딕셔너리
        """
        # 입력이 시리즈인 경우 numpy 배열로 변환
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
            
        # 차원 확인 및 조정 (1D array로 변환)
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        
        # 길이 확인
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true has length {len(y_true)}, y_pred has length {len(y_pred)}")
            
        # 정확도 계산 (정확히 일치하는 비율)
        accuracy = np.mean(y_true == y_pred)
        
        # F1 점수 계산 (가중 평균 사용)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # 혼동 행렬 계산
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
        
        # 분류 보고서 생성
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=['-1', '0', '1'],
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': report
        }
    
    def save_model(self, dir_path: str) -> None:
        """
        모델과 메타데이터를 저장합니다.
        
        Args:
            dir_path (str): 모델을 저장할 디렉토리 경로
        """
        # 디렉토리 생성
        os.makedirs(dir_path, exist_ok=True)
        
        # 모델 저장
        model_path = os.path.join(dir_path, 'model.txt')
        self.model.save_model(model_path)
        
        # 메타데이터 저장
        metadata = {
            'best_params': self.best_params,
            'feature_importances': self.feature_importances_.to_dict() if self.feature_importances_ is not None else {},
            'feature_columns': list(self.feature_importances_.index) if self.feature_importances_ is not None else []
        }
        
        with open(os.path.join(dir_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {dir_path}")
    
    @classmethod
    def load_model(cls, dir_path: str) -> 'ModelTrainer':
        """
        저장된 모델을 로드합니다.
        
        Args:
            dir_path (str): 모델이 저장된 디렉토리 경로
            
        Returns:
            ModelTrainer: 로드된 모델 인스턴스
        """
        # 모델 인스턴스 생성
        trainer = cls()
        
        # 모델 로드
        model_path = os.path.join(dir_path, 'model.txt')
        trainer.model = lgb.Booster(model_file=model_path)
        
        # 메타데이터 로드
        with open(os.path.join(dir_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        trainer.best_params = metadata.get('best_params', {})
        if 'feature_importances' in metadata and metadata['feature_importances']:
            trainer.feature_importances_ = pd.Series(metadata['feature_importances'])
        
        return trainer

# 사용 예시
if __name__ == "__main__":
    # 예제 데이터 생성
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.choice([-1, 0, 1], size=1000))
    
    # 모델 훈련
    trainer = ModelTrainer()
    metrics = trainer.train(X, y, n_splits=3, n_trials=10, optimize=True)
    
    # 모델 저장
    trainer.save_model('models/lightgbm_model')
    
    # 모델 로드
    loaded_trainer = ModelTrainer.load_model('models/lightgbm_model')
    predictions = loaded_trainer.predict(X)
    print(f"Predictions: {predictions[:10]}...")
