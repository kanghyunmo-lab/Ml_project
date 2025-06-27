"""
모델 예측을 위한 유틸리티 함수들을 포함하는 모듈입니다.
"""
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

def load_model(model_path: Union[str, Path]):
    """저장된 모델을 로드합니다.
    
    Args:
        model_path: 모델 파일 경로
        
    Returns:
        로드된 모델 객체
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"모델이 성공적으로 로드되었습니다: {model_path}")
        return model
    except Exception as e:
        logger.error(f"모델 로드 중 오류가 발생했습니다: {e}")
        raise

def save_model(model, model_path: Union[str, Path]):
    """모델을 저장합니다.
    
    Args:
        model: 저장할 모델 객체
        model_path: 저장할 파일 경로
    """
    try:
        # 디렉토리가 없는 경우 생성
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        joblib.dump(model, model_path)
        logger.info(f"모델이 성공적으로 저장되었습니다: {model_path}")
    except Exception as e:
        logger.error(f"모델 저장 중 오류가 발생했습니다: {e}")
        raise

def predict(model, X: pd.DataFrame) -> np.ndarray:
    """입력 데이터에 대한 예측을 수행합니다.
    
    Args:
        model: 예측에 사용할 모델
        X: 입력 특성 데이터프레임
        
    Returns:
        예측 결과 배열
    """
    try:
        predictions = model.predict(X)
        logger.info(f"예측이 완료되었습니다. 예측 결과: {predictions[:5]}...")
        return predictions
    except Exception as e:
        logger.error(f"예측 중 오류가 발생했습니다: {e}")
        raise

def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """각 클래스에 대한 확률을 예측합니다.
    
    Args:
        model: 예측에 사용할 모델 (predict_proba 메서드를 가진 모델이어야 함)
        X: 입력 특성 데이터프레임
        
    Returns:
        각 클래스에 대한 확률 배열
    """
    try:
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            logger.info("확률 예측이 완료되었습니다.")
            return probas
        else:
            raise AttributeError("이 모델은 확률 예측을 지원하지 않습니다.")
    except Exception as e:
        logger.error(f"확률 예측 중 오류가 발생했습니다: {e}")
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """모델 성능을 평가합니다.
    
    Args:
        model: 평가할 모델
        X_test: 테스트 특성 데이터프레임
        y_test: 테스트 타겟 시리즈
        
    Returns:
        평가 지표를 포함한 딕셔너리
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    try:
        # 예측 수행
        y_pred = model.predict(X_test)
        
        # 평가 지표 계산
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # 확률 예측이 가능한 경우 ROC AUC도 계산
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] > 2:  # 다중 클래스인 경우
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
                else:  # 이진 분류인 경우
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            except Exception as e:
                logger.warning(f"ROC AUC 계산 중 오류가 발생했습니다: {e}")
        
        logger.info("모델 평가가 완료되었습니다.")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return metrics
        
    except Exception as e:
        logger.error(f"모델 평가 중 오류가 발생했습니다: {e}")
        raise
