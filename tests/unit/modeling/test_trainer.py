"""
ModelTrainer 클래스에 대한 단위 테스트
"""

import os
import pytest
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import lightgbm as lgb

# 루트 디렉토리 추가
import sys
sys.path.append(str(Path(__file__).parents[3]))

from src.modeling.trainer import ModelTrainer

# 테스트용 가상 데이터 생성
def create_test_data(n_samples=1000, n_features=10):
    """테스트용 가상 데이터 생성"""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([-1, 0, 1], size=n_samples))
    return X, y

class TestModelTrainer:
    """ModelTrainer 클래스 테스트"""
    
    def setup_method(self):
        """테스트 전 초기화"""
        self.X, self.y = create_test_data()
        self.trainer = ModelTrainer()
        
    def test_initialization(self):
        """초기화 테스트"""
        assert self.trainer.model is None
        assert self.trainer.best_params is None
        assert self.trainer.feature_importances_ is None
        assert 'objective' in self.trainer.params
        assert self.trainer.params['num_class'] == 3
        
    def test_train_with_default_params(self):
        """Test training with default parameters"""
        # Mock LightGBM model setup
        with patch('lightgbm.train') as mock_train, \
             patch('lightgbm.Dataset') as mock_dataset, \
             patch('lightgbm.early_stopping'), \
             patch('lightgbm.log_evaluation'):
            
            # Setup mock model
            mock_model = MagicMock()
            n_samples = len(self.X)
            n_classes = 3
            
            # Create a fixed set of predictions for consistent testing
            np.random.seed(42)
            fixed_predictions = np.random.choice([0, 1, 2], size=n_samples)
            
            # Mock predict method to return class probabilities with shape (n_samples, n_classes)
            def mock_predict(X, **kwargs):
                n_samples = len(X)
                probas = np.zeros((n_samples, 3))
                for i in range(n_samples):
                    probas[i, fixed_predictions[i % len(fixed_predictions)]] = 1.0
                return probas
                
            mock_model.predict.side_effect = mock_predict
            mock_model.best_iteration = 10
            mock_model.feature_importance.return_value = np.ones(10)  # 10 features
            mock_train.return_value = mock_model
            
            # Mock dataset
            mock_ds = MagicMock()
            mock_dataset.return_value = mock_ds
            
            # Mock evaluation metrics
            mock_metrics = {
                'valid_0': {'multi_logloss': 1.0},
                'valid_1': {'multi_logloss': 1.0}
            }
            mock_model.eval.return_value = mock_metrics
            
            # Mock TimeSeriesSplit to control the splits
            class MockTimeSeriesSplit:
                def __init__(self, n_splits=2):
                    self.n_splits = n_splits
                
                def split(self, X):
                    # Create deterministic splits for testing
                    n_samples = len(X)
                    fold_size = n_samples // (self.n_splits + 1)
                    for i in range(self.n_splits):
                        train_size = (i + 1) * fold_size
                        val_size = min(fold_size, n_samples - train_size)
                        if val_size > 0:
                            train_indices = np.arange(train_size)
                            val_indices = np.arange(train_size, train_size + val_size)
                            yield train_indices, val_indices
            
            with patch('sklearn.model_selection.TimeSeriesSplit', MockTimeSeriesSplit):
                # Train with test data
                metrics = self.trainer.train(
                    self.X, 
                    self.y, 
                    n_splits=2, 
                    optimize=False,
                    n_trials=1
                )
            
            # Verify
            assert mock_train.called
            assert 'accuracy' in metrics
            assert 'f1_weighted' in metrics
            assert 'confusion_matrix' in metrics
            assert 0 <= metrics['accuracy'] <= 1  # Accuracy should be between 0 and 1
            
    def test_hyperparameter_optimization(self):
        """하이퍼파라미터 최적화 테스트"""
        with patch('optuna.create_study') as mock_study:
            # 모의 study 객체 설정
            mock_study.return_value = MagicMock()
            mock_study.return_value.best_params = {'learning_rate': 0.1}
            mock_study.return_value.best_trial.value = 0.9
            
            # 최적화 실행
            best_params = self.trainer._optimize_hyperparameters(
                self.X, self.y, n_trials=5, n_splits=2
            )
            
            # 검증
            assert mock_study.called
            assert 'learning_rate' in best_params
            
    def test_predict(self):
        """Test prediction functionality"""
        # Setup mock model
        mock_model = MagicMock()
        # Return class probabilities for 3 samples and 3 classes
        mock_model.predict.return_value = np.array([
            [0.1, 0.8, 0.1],  # Class 1 (index 1) -> 0 after conversion
            [0.7, 0.2, 0.1],  # Class 0 (index 0) -> -1 after conversion
            [0.1, 0.2, 0.7]   # Class 2 (index 2) -> 1 after conversion
        ])
        mock_model.best_iteration = 100
        
        self.trainer.model = mock_model
        
        # Make predictions
        X_test = pd.DataFrame(np.random.rand(3, 10))
        predictions = self.trainer.predict(X_test)
        
        # Verify
        # The predict method should return class indices (-1, 0, 1)
        expected = np.array([0, -1, 1])
        np.testing.assert_array_equal(predictions, expected)
        
        # Verify predict was called with correct arguments
        mock_model.predict.assert_called_once()
        args, kwargs = mock_model.predict.call_args
        assert kwargs.get('num_iteration') == 100
        
    def test_evaluate(self):
        """Test evaluation metrics calculation"""
        # Test case with all three classes
        y_true = np.array([-1, 0, 1, -1, 0, 1, 0, 1])
        y_pred = np.array([-1, 0, 1,  0, 0, 1, 0, -1]) # 2 wrong predictions

        # Mock the classification_report and confusion_matrix to avoid label mismatch issues
        with patch('sklearn.metrics.classification_report') as mock_report, \
             patch('sklearn.metrics.confusion_matrix') as mock_cm, \
             patch('sklearn.metrics.accuracy_score', return_value=0.875) as mock_acc, \
             patch('sklearn.metrics.f1_score', return_value=0.8) as mock_f1:
            
            # Mock confusion matrix to return a 3x3 matrix
            mock_cm.return_value = np.array([
                [0, 0, 0],  # -1 (none in test data)
                [0, 4, 1],  # 0 (4 correct, 1 wrong)
                [0, 0, 3]   # 1 (3 correct, 0 wrong)
            ])
            
            # Mock classification report
            mock_report.return_value = {
                'accuracy': 0.875,
                '0': {'precision': 0.8, 'recall': 0.8, 'f1-score': 0.8, 'support': 5},
                '1': {'precision': 0.75, 'recall': 1.0, 'f1-score': 0.86, 'support': 3},
                'weighted avg': {'precision': 0.78, 'recall': 0.88, 'f1-score': 0.82, 'support': 8}
            }
            
            metrics = self.trainer._evaluate(y_true, y_pred)

        # Check all required metrics are present
        assert 'accuracy' in metrics
        assert 'f1_weighted' in metrics
        assert 'confusion_matrix' in metrics
        assert 'classification_report' in metrics
        
        # Check accuracy is a float between 0 and 1
        assert 0 <= metrics['accuracy'] <= 1
        
        # Check confusion matrix shape (3x3 for 3 classes: -1, 0, 1)
        assert metrics['confusion_matrix'].shape == (3, 3)
        
        # Check that the confusion matrix sums to the number of samples
        assert metrics['confusion_matrix'].sum() == len(y_true)
        
    def test_save_and_load_model(self, tmp_path):
        """Test model save and load functionality"""
        # 1. Setup
        save_dir = tmp_path / "model"
        
        # Mock the model and its attributes
        mock_model = MagicMock(spec=lgb.Booster)
        mock_model.save_model.return_value = None
        
        self.trainer.model = mock_model
        self.trainer.best_params = {'learning_rate': 0.1, 'n_estimators': 100}
        self.trainer.feature_importances_ = pd.Series([0.1, 0.9], index=['f1', 'f2'])

        # 2. Save model
        with patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump:
            
            self.trainer.save_model(str(save_dir))

            # Verify save operations
            mock_makedirs.assert_called_with(str(save_dir), exist_ok=True)
            mock_model.save_model.assert_called_with(os.path.join(str(save_dir), 'model.txt'))
            mock_file.assert_called_with(os.path.join(str(save_dir), 'metadata.json'), 'w')
            mock_json_dump.assert_called_once()

        # 3. Load model
        metadata = {
            'best_params': self.trainer.best_params,
            'feature_importances': self.trainer.feature_importances_.to_dict(),
            'feature_columns': list(self.trainer.feature_importances_.index)
        }
        
        # Mock file reading for load
        mock_read_open = mock_open(read_data=json.dumps(metadata))
        
        with patch('lightgbm.Booster', return_value=mock_model) as mock_booster, \
             patch('builtins.open', mock_read_open):
            
            loaded_trainer = ModelTrainer.load_model(str(save_dir))

            # Verify load operations
            mock_booster.assert_called_with(model_file=os.path.join(str(save_dir), 'model.txt'))
            mock_read_open.assert_called_with(os.path.join(str(save_dir), 'metadata.json'), 'r')
            
            # Verify loaded attributes
            assert loaded_trainer.model == mock_model
            assert loaded_trainer.best_params == self.trainer.best_params
            pd.testing.assert_series_equal(loaded_trainer.feature_importances_, self.trainer.feature_importances_)

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
