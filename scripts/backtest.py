"""
암호화폐 트레이딩 전략 백테스트 스크립트

기능:
1. 훈련된 LightGBM 모델 로드
2. 백테스트 데이터 준비 및 전처리
3. 백테스트 실행
4. 결과 분석 및 시각화
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import joblib

# 프로젝트 루트 디렉토리 추가
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest.log')
    ]
)
logger = logging.getLogger(__name__)

# 프로젝트 모듈 임포트
try:
    from src.modeling.trainer import ModelTrainer
    from src.processing.feature_engineer import FeatureEngineer
    from src.utils.config_loader import load_config
except ImportError as e:
    logger.error(f"필요한 모듈을 임포트하는 중 오류 발생: {e}")
    sys.exit(1)

class BacktestEngine:
    """백테스트 엔진 클래스"""
    
    def __init__(self, data, initial_capital=10000, commission=0.001, force_signal: Optional[Dict] = None):
        """
        백테스트 엔진 초기화
        
        Args:
            data: 백테스트용 데이터 (DataFrame)
            initial_capital: 초기 자본 (USDT)
            commission: 거래 수수료 (0.001 = 0.1%)
            force_signal: 강제로 주입할 신호 (테스트용)
                - 'signal': 고정 신호 값 (-1, 0, 1)
                - 'start_date': 적용 시작일 (포함)
                - 'end_date': 적용 종료일 (미포함)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.force_signal = force_signal
        self.results = None
        
        # 신호 분포 검증
        if 'prediction' in self.data.columns:
            self._validate_signal_distribution()
    
    def _validate_signal_distribution(self):
        """예측 신호의 분포를 검증하고 로깅합니다."""
        if 'prediction' not in self.data.columns:
            return
            
        signals = self.data['prediction']
        signal_counts = signals.value_counts().sort_index()
        total = len(signals)
        
        logger.info("Signal distribution in backtest data:")
        for signal, count in signal_counts.items():
            logger.info(f"  Signal {signal}: {count} samples ({count/total:.1%})")
        
        # 모든 신호가 0인 경우 경고
        if len(signal_counts) == 1 and 0 in signal_counts:
            logger.warning("All signals are 0 (HOLD). No trades will be executed.")
    
    def _apply_force_signal(self, current_time, current_signal):
        """강제 신호를 적용합니다."""
        if not self.force_signal:
            return current_signal
            
        signal = self.force_signal.get('signal', 0)
        start_date = pd.to_datetime(self.force_signal.get('start_date', '1970-01-01'))
        end_date = pd.to_datetime(self.force_signal.get('end_date', '2100-01-01'))
        
        if start_date <= current_time < end_date:
            logger.debug(f"Applying forced signal {signal} at {current_time}")
            return signal
            
        return current_signal
    
    def run(self):
        """백테스트 실행"""
        logger.info("백테스트 실행 중...")
        
        # 초기화
        self.data['position'] = 0  # 0: 홀드, 1: 롱, -1: 숏
        self.data['equity'] = self.initial_capital
        self.data['returns'] = 0.0
        
        cash = self.initial_capital
        position = 0
        entry_price = 0
        
        # 백테스트 실행
        for i in range(1, len(self.data)):
            current = self.data.iloc[i]
            prev = self.data.iloc[i-1]
            current_time = current.name if hasattr(current, 'name') else self.data.index[i]
            
            # 1. 신호 가져오기
            signal = current.get('prediction', 0)
            
            # 2. 강제 신호 적용 (테스트용)
            signal = self._apply_force_signal(current_time, signal)
            
            # 3. 포지션 업데이트
            # 포지션 종료 조건 (반대 신호 또는 청산 신호)
            if position != 0 and signal != position and signal != 0:  # 0(홀드)이 아닌 반대 신호인 경우에만 종료
                # 수익률 계산 (수수료 고려)
                if position == 1:  # 롱 포지션 종료
                    returns = (current['close'] / entry_price - 1) * (1 - self.commission)
                    logger.debug(f"롱 포지션 종료: 진입가 {entry_price:.2f}, 종가 {current['close']:.2f}, 수익률: {returns*100:.2f}%")
                else:  # 숏 포지션 종료
                    returns = (1 - current['close'] / entry_price) * (1 - self.commission)
                    logger.debug(f"숏 포지션 종료: 진입가 {entry_price:.2f}, 종가 {current['close']:.2f}, 수익률: {returns*100:.2f}%")
                
                cash *= (1 + returns)
                position = 0
                entry_price = 0
            
            # 신규 포지션 진입 (현재 포지션이 없고, 신호가 0이 아닌 경우)
            if position == 0 and signal != 0:
                position = signal
                entry_price = current['close']
                logger.debug(f"{'롱' if position == 1 else '숏'} 포지션 진입: 가격 {entry_price:.2f}, 잔고: {cash:.2f} USDT")
            
            # 잔고 업데이트
            self.data.iloc[i, self.data.columns.get_loc('position')] = position
            self.data.iloc[i, self.data.columns.get_loc('equity')] = cash
            
            # 일별 수익률 계산
            if position == 1:  # 롱 포지션
                self.data.iloc[i, self.data.columns.get_loc('returns')] = (current['close'] / prev['close'] - 1) * (1 - self.commission)
            elif position == -1:  # 숏 포지션
                self.data.iloc[i, self.data.columns.get_loc('returns')] = (1 - current['close'] / prev['close']) * (1 - self.commission)
            else:  # 홀드
                self.data.iloc[i, self.data.columns.get_loc('returns')] = 0
        
        # 결과 저장
        self.results = {
            'equity': self.data['equity'].values,
            'returns': self.data['returns'].values,
            'positions': self.data['position'].values,
            'trades': self.extract_trades()  # 거래 내역 추가
        }
        
        return self.results
    
    def extract_trades(self):
        """백테스트 엔진에서 거래 내역을 추출합니다."""
        trades = []
        position = 0
        entry_idx = 0
        entry_price = 0
        
        for i in range(1, len(self.data)):
            current = self.data.iloc[i]
            prev = self.data.iloc[i-1]
            
            # 포지션 진입
            if position == 0 and current['position'] != 0:
                position = current['position']
                entry_idx = i
                entry_price = current['close']
            # 포지션 종료
            elif position != 0 and current['position'] != position:
                exit_price = current['close']
                if position == 1:  # 롱 포지션 종료
                    pnl = (exit_price / entry_price - 1) * (1 - self.commission)
                else:  # 숏 포지션 종료
                    pnl = (1 - exit_price / entry_price) * (1 - self.commission)
                
                trades.append({
                    'entry_time': self.data.index[entry_idx],
                    'exit_time': self.data.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': 1.0,  # 고정 크기
                    'pnl': pnl * self.initial_capital * 0.02,  # 2% 포지션 크기 가정
                    'pnl_pct': pnl * 100,
                    'type': 'LONG' if position == 1 else 'SHORT',
                    'duration': (self.data.index[i] - self.data.index[entry_idx]).total_seconds() / 3600  # 시간 단위
                })
                
                position = current['position']
                if position != 0:  # 반대 방향으로 즉시 진입
                    entry_idx = i
                    entry_price = exit_price
        
        return trades

class BacktestRunner:
    """백테스트 실행 클래스"""
    
    def __init__(self, config_path: str = 'config/backtest_config.yaml'):
        """
        백테스트 실행기 초기화
        
        Args:
            config_path: 백테스트 설정 파일 경로
        """
        self.config = load_config(config_path)
        self.model = None
        self.data = None
        self.results = None
        
    def load_model(self):
        """훈련된 모델 로드"""
        try:
            model_path = self.config['model_path']
            logger.info(f"모델 로드 중: {model_path}")
            
            # LightGBM 부스터로 모델 로드
            self.model = lgb.Booster(model_file=model_path)
            logger.info("모델 로드 완료")
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    def prepare_data(self):
        """백테스트용 데이터 준비"""
        try:
            data_path = self.config['data_path']
            logger.info(f"데이터 로드 중: {data_path}")
            
            # 데이터 로드
            self.data = pd.read_parquet(data_path)
            
            # 모델이 예상하는 특성만 선택 (훈련 시 사용된 순서대로 정렬)
            expected_features = [
                'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 
                'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 
                'bb_middle', 'bb_lower', 'bb_width', 'atr', 'volume_ma_20', 
                'volume_momentum', 'obv', 'obv_ema', 'vpt', 'volume_volatility'
            ]
            
            # 데이터에서 필요한 특성만 선택
            available_features = [col for col in expected_features if col in self.data.columns]
            missing_features = set(expected_features) - set(available_features)
            
            if missing_features:
                logger.warning(f"다음 특성들을 찾을 수 없습니다: {missing_features}")
                
            self.features = self.data[available_features]
            
            # 특성 순서를 모델이 예상하는 순서대로 정렬
            self.features = self.features[expected_features if not missing_features else available_features]
            
            logger.info(f"데이터 로드 완료. 행 수: {len(self.data)}")
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {e}")
            raise
    
    def run_backtest(self):
        """백테스트 실행"""
        try:
            logger.info("백테스트 시작...")
            
            # 예측 수행 (확률 예측)
            logger.info("예측 수행 중...")
            
            # LightGBM Booster를 사용한 예측 (클래스별 확률 예측)
            # predict 메서드는 각 행에 대한 클래스 확률을 반환 (n_samples x n_classes)
            proba_predictions = self.model.predict(self.features, num_iteration=self.model.best_iteration, predict_disable_shape_check=True)
            
            # 예측값 로깅
            logger.info(f"예측값 통계 - 평균: {np.mean(proba_predictions):.4f}, 최소: {np.min(proba_predictions):.4f}, 최대: {np.max(proba_predictions):.4f}")
            
            # 예측값을 -1, 0, 1로 변환 (가장 높은 확률을 가진 클래스 선택)
            # 클래스는 0: 매도, 1: 홀드, 2: 매수로 가정
            self.data['prediction'] = 0  # 기본값: 홀드
            
            # 가장 높은 확률을 가진 클래스 인덱스 찾기
            if proba_predictions.ndim > 1:  # 다중 클래스 예측인 경우
                # 각 행에서 가장 높은 확률을 가진 클래스의 인덱스를 찾음
                predicted_classes = np.argmax(proba_predictions, axis=1)
                
                # 클래스 인덱스를 -1, 0, 1로 변환 (0: 매도, 1: 홀드, 2: 매수 -> -1: 매도, 0: 홀드, 1: 매수)
                self.data['prediction'] = predicted_classes - 1
                
                # 예측 분포 로깅
                unique, counts = np.unique(self.data['prediction'], return_counts=True)
                pred_dist = dict(zip(unique, counts))
                logger.info(f"예측 분포: {pred_dist}")
                
                # 각 클래스별 확률 분포 확인
                logger.info(f"매도 확률 - 평균: {np.mean(proba_predictions[:, 0]):.4f}, 최대: {np.max(proba_predictions[:, 0]):.4f}")
                logger.info(f"홀드 확률 - 평균: {np.mean(proba_predictions[:, 1]):.4f}, 최대: {np.max(proba_predictions[:, 1]):.4f}")
                logger.info(f"매수 확률 - 평균: {np.mean(proba_predictions[:, 2]):.4f}, 최대: {np.max(proba_predictions[:, 2]):.4f}")
                
                # 임계값을 조정하여 더 많은 신호 생성 (테스트용)
                # 매수 신호: 매수 확률이 홀드보다 10%p 이상 높은 경우
                buy_condition = (proba_predictions[:, 2] > (proba_predictions[:, 1] + 0.1))
                # 매도 신호: 매도 확률이 홀드보다 10%p 이상 높은 경우
                sell_condition = (proba_predictions[:, 0] > (proba_predictions[:, 1] + 0.1))
                
                self.data.loc[buy_condition, 'prediction'] = 1
                self.data.loc[sell_condition, 'prediction'] = -1
                
                # 조정 후 예측 분포 로깅
                unique, counts = np.unique(self.data['prediction'], return_counts=True)
                pred_dist = dict(zip(unique, counts))
                logger.info(f"조정 후 예측 분포: {pred_dist}")
            else:  # 회귀 또는 이진 분류인 경우
                # 회귀값을 -1과 1 사이로 정규화 (임계값: -0.2, 0.2)
                threshold = 0.2
                self.data.loc[proba_predictions > threshold, 'prediction'] = 1    # 매수
                self.data.loc[proba_predictions < -threshold, 'prediction'] = -1  # 매도
                
                # 예측 분포 로깅
                buy_signals = (self.data['prediction'] == 1).sum()
                sell_signals = (self.data['prediction'] == -1).sum()
                hold_signals = (self.data['prediction'] == 0).sum()
                logger.info(f"매수 신호: {buy_signals}, 매도 신호: {sell_signals}, 홀드: {hold_signals}")
            
            # 예측값 샘플 로깅 (처음 5개 행)
            logger.info("예측값 샘플 (처음 5개 행):")
            for i in range(min(5, len(proba_predictions))):
                if proba_predictions.ndim > 1:
                    logger.info(f"  행 {i}: 매도={proba_predictions[i, 0]:.4f}, 홀드={proba_predictions[i, 1]:.4f}, 매수={proba_predictions[i, 2]:.4f} -> 예측={self.data['prediction'].iloc[i]}")
                else:
                    logger.info(f"  행 {i}: 예측값={proba_predictions[i]:.4f} -> 신호={self.data['prediction'].iloc[i]}")
            
            # 백테스트 엔진 초기화
            engine = BacktestEngine(
                data=self.data,
                initial_capital=self.config.get('initial_capital', 10000),
                commission=self.config.get('commission', 0.001)
            )
            
            # 백테스트 실행
            logger.info("백테스트 실행 중...")
            engine.run()
            
            # 결과 저장
            self.results = engine.results
            self.data['equity'] = self.results['equity']
            self.data['returns'] = self.results['returns']
            self.data['position'] = self.results['positions']
            
            # 결과에서 거래 내역 추출
            self.trades = self.results.get('trades', [])
            logger.info(f"백테스트 완료. 총 {len(self.trades)}건의 거래 발생")
            
        except Exception as e:
            logger.error(f"백테스트 실행 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def analyze_results(self):
        """백테스트 결과 분석"""
        if self.results is None:
            logger.error("백테스트를 먼저 실행해주세요.")
            return
        
        try:
            # 기본 통계 계산
            equity = self.data['equity']
            returns = self.data['returns']
            
            total_return = equity.iloc[-1] / equity.iloc[0] - 1
            max_drawdown = self.calculate_max_drawdown(equity)
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            
            # 거래 통계
            trades = self.extract_trades()
            win_rate = (trades['pnl'] > 0).mean() * 100 if len(trades) > 0 else 0
            avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if (trades['pnl'] > 0).any() else 0
            avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if (trades['pnl'] < 0).any() else 0
            profit_factor = -trades[trades['pnl'] > 0]['pnl'].sum() / trades[trades['pnl'] < 0]['pnl'].sum() if (trades['pnl'] < 0).any() else float('inf')
            
            logger.info("\n=== 백테스트 결과 ===")
            logger.info(f"초기 자본: ${equity.iloc[0]:,.2f}")
            logger.info(f"최종 자본: ${equity.iloc[-1]:,.2f}")
            logger.info(f"총 수익률: {total_return*100:.2f}%")
            logger.info(f"최대 낙폭: {max_drawdown*100:.2f}%")
            logger.info(f"샤프 지수: {sharpe_ratio:.2f}")
            logger.info(f"\n=== 거래 통계 ===")
            logger.info(f"총 거래 횟수: {len(trades)}")
            logger.info(f"승률: {win_rate:.2f}%")
            logger.info(f"평균 수익: {avg_win*100:.2f}%")
            logger.info(f"평균 손실: {avg_loss*100:.2f}%")
            logger.info(f"프로핏 팩터: {profit_factor:.2f}")
            
            # 결과 시각화
            self.plot_results()
            
            return {
                'initial_capital': equity.iloc[0],
                'final_equity': equity.iloc[-1],
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'num_trades': len(trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"결과 분석 중 오류 발생: {e}")
            raise
    
    def extract_trades(self):
        """거래 내역 추출"""
        if not hasattr(self, 'data') or 'position' not in self.data.columns:
            logger.error("백테스트를 먼저 실행해주세요.")
            return pd.DataFrame()
        
        try:
            trades = []
            position = 0
            entry_idx = 0
            entry_price = 0
            
            for i in range(1, len(self.data)):
                current = self.data.iloc[i]
                prev = self.data.iloc[i-1]
                
                # 포지션 진입
                if position == 0 and current['position'] != 0:
                    position = current['position']
                    entry_idx = i
                    entry_price = current['close']
                # 포지션 종료
                elif position != 0 and current['position'] != position:
                    exit_price = current['close']
                    if position == 1:  # 롱 포지션 종료
                        pnl = (exit_price / entry_price - 1) * (1 - self.config.get('commission', 0.001))
                    else:  # 숏 포지션 종료
                        pnl = (1 - exit_price / entry_price) * (1 - self.config.get('commission', 0.001))
                    
                    trades.append({
                        'entry_time': self.data.index[entry_idx],
                        'exit_time': self.data.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'size': 1.0,  # 고정 크기
                        'pnl': pnl * self.config.get('initial_capital', 10000) * 0.02,  # 2% 포지션 크기 가정
                        'pnl_pct': pnl * 100,
                        'type': 'LONG' if position == 1 else 'SHORT',
                        'duration': (self.data.index[i] - self.data.index[entry_idx]).total_seconds() / 3600  # 시간 단위
                    })
                    
                    position = current['position']
                    if position != 0:  # 반대 방향으로 즉시 진입
                        entry_idx = i
                        entry_price = exit_price
            
            if trades:
                return pd.DataFrame(trades)
            else:
                logger.warning("거래 내역이 없습니다.")
                return pd.DataFrame(columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'size', 'pnl', 'pnl_pct', 'type', 'duration'])
                
        except Exception as e:
            logger.error(f"거래 내역 추출 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    @staticmethod
    def calculate_max_drawdown(equity_curve):
        """최대 낙폭 계산"""
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=365*24/4):  # 4시간 봉 기준
        """샤프 지수 계산"""
        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * excess_returns.mean() / (returns.std() + 1e-10)
    
    def plot_results(self):
        """백테스트 결과 시각화"""
        try:
            plt.figure(figsize=(14, 10))
            
            # 1. 자산 곡선
            plt.subplot(3, 1, 1)
            plt.plot(self.data.index, self.data['equity'], label='Equity', color='blue')
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            
            # 롱/숏 포지션 표시
            long_positions = self.data[self.data['position'] == 1]
            short_positions = self.data[self.data['position'] == -1]
            
            if not long_positions.empty:
                plt.scatter(long_positions.index, 
                           long_positions['equity'], 
                           color='green', marker='^', alpha=0.7, label='Long')
            if not short_positions.empty:
                plt.scatter(short_positions.index, 
                           short_positions['equity'], 
                           color='red', marker='v', alpha=0.7, label='Short')
            
            plt.legend()
            
            # 2. 일별 수익률
            plt.subplot(3, 1, 2)
            plt.bar(self.data.index, self.data['returns'] * 100, width=1.0, color=np.where(self.data['returns'] >= 0, 'g', 'r'), alpha=0.7)
            plt.title('Daily Returns')
            plt.xlabel('Date')
            plt.ylabel('Return (%)')
            plt.grid(True)
            
            # 3. 가격 차트와 포지션
            plt.subplot(3, 1, 3)
            plt.plot(self.data.index, self.data['close'], label='Price', color='black')
            
            # 롱/숏 진입점 표시
            if not long_positions.empty:
                plt.scatter(long_positions.index, 
                           long_positions['close'], 
                           color='green', marker='^', alpha=0.7, label='Long')
            if not short_positions.empty:
                plt.scatter(short_positions.index, 
                           short_positions['close'], 
                           color='red', marker='v', alpha=0.7, label='Short')
            
            plt.title('Price with Trades')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.grid(True)
            plt.legend()
            
            # 결과 저장
            os.makedirs('results', exist_ok=True)
            plot_path = 'results/backtest_results.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
            logger.info(f"결과 그래프 저장 완료: {plot_path}")
            
        except Exception as e:
            logger.error(f"결과 시각화 중 오류 발생: {e}")
            raise

def main():
    """메인 실행 함수"""
    try:
        logger.info("=== 백테스트 시작 ===")
        
        # 백테스트 실행기 초기화
        backtest = BacktestRunner()
        
        # 모델 로드
        backtest.load_model()
        
        # 데이터 준비
        backtest.prepare_data()
        
        # 백테스트 실행
        backtest.run_backtest()
        
        # 결과 분석
        backtest.analyze_results()
        
        logger.info("=== 백테스트 완료 ===")
        return 0
    except Exception as e:
        logger.error(f"백테스트 실행 중 치명적 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # GUI 백엔드 사용 방지
    sys.exit(main())
import matplotlib.pyplot as plt
from pathlib import Path
import json
import joblib
import traceback
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest.log')
    ]
)
logger = logging.getLogger(__name__)

from src.backtesting.engine import BacktestEngine, TripleBarrierStrategy

# 상수 정의
DATA_DIR = Path('data/processed')  # 데이터 디렉토리
MODELS_DIR = Path('models')        # 모델 저장 디렉토리
REPORT_DIR = Path('results/reports')  # 보고서 저장 디렉토리
PLOT_DIR = Path('results/plots')    # 차트 저장 디렉토리

# 필요한 디렉토리 생성
for directory in [DATA_DIR, MODELS_DIR, REPORT_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def load_data_and_predict(force_signal: Optional[Dict] = None):
    """
    데이터를 로드하고 훈련된 모델로 예측을 수행합니다.
    
    Args:
        force_signal: 강제로 주입할 신호 (테스트용)
            - 'signal': 고정 신호 값 (-1, 0, 1)
            - 'start_date': 적용 시작일 (포함)
            - 'end_date': 적용 종료일 (미포함)
            
    Returns:
        pd.DataFrame: 예측 결과가 추가된 데이터프레임
    """
    logger.info("데이터 로드 및 예측 시작...")
    
    try:
        # 1. 실제 데이터 로드
        data_path = DATA_DIR / 'btc_usdt_4h_with_target.parquet'
        if not data_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
            
        logger.info(f"데이터 로드 중: {data_path}")
        data = pd.read_parquet(data_path)
        
        # 2. 강제 신호가 주어진 경우, 예측 단계를 건너뛰고 바로 강제 신호 적용
        if force_signal:
            logger.info("강제 신호 모드: 모델 예측을 건너뜁니다.")
            data['prediction'] = 0  # 기본값으로 0(홀드) 설정
            
            start_date = pd.to_datetime(force_signal.get('start_date', '1970-01-01'))
            end_date = pd.to_datetime(force_signal.get('end_date', '2100-01-01'))
            signal_value = force_signal.get('signal', 0)
            
            # 날짜 범위에 맞는 행 선택
            mask = (data['datetime'] >= start_date) & (data['datetime'] < end_date)
            data.loc[mask, 'prediction'] = signal_value
            
            logger.info(f"강제 신호 적용: {start_date} ~ {end_date} 구간에 신호 {signal_value} 적용")
            logger.info(f"강제 신호 적용된 행 수: {mask.sum()} / {len(data)}")
        else:
            # 3. 훈련된 모델 로드
            model_path = MODELS_DIR / 'lightgbm_model'
            if not model_path.exists():
                raise FileNotFound(f"모델 파일을 찾을 수 없습니다: {model_path}")
                
            logger.info(f"모델 로드 중: {model_path}")
            model = joblib.load(model_path)
            
            # 4. 피처 엔지니어링
            from src.processing.feature_engineer import FeatureEngineer
            feature_engineer = FeatureEngineer(data)
            features = feature_engineer.add_technical_indicators()
            
            # 5. 예측 수행
            logger.info("예측 수행 중...")
            X = features.drop(columns=['open', 'high', 'low', 'close', 'volume', 'target'], errors='ignore')
            
            # 모델 예측 (예측 확률 포함)
            data['prediction'], pred_proba = model.predict(X, return_proba=True, verbose=True)
            
            # 예측 확률 추가
            for i, cls in enumerate([-1, 0, 1]):
                data[f'prob_{cls}'] = pred_proba[:, i]
        
        # 예측 분포 로깅
        signal_dist = data['prediction'].value_counts().sort_index()
        logger.info("예측 신호 분포:")
        for signal, count in signal_dist.items():
            logger.info(f"  신호 {signal}: {count}건 ({(count/len(data))*100:.1f}%)")
            
            # 신호가 0뿐인 경우 경고
            if len(signal_dist) == 1 and 0 in signal_dist:
                logger.warning("모든 신호가 0(홀드)입니다. 거래가 발생하지 않을 수 있습니다.")
        
        # 6. 결과 저장
        output_path = DATA_DIR / 'backtest_data.parquet'
        data.to_parquet(output_path)
        logger.info(f"예측 완료. 데이터 저장됨: {output_path}")
        
        return data
        
    except Exception as e:
        logger.error(f"데이터 로드 및 예측 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def run_backtest(data: pd.DataFrame, force_signal: Optional[Dict] = None):
    """
    백테스트를 실행하고 결과를 반환합니다.
    
    Args:
        data: 백테스트에 사용할 데이터프레임
        force_signal: 강제로 주입할 신호 (테스트용)
            - 'signal': 고정 신호 값 (-1, 0, 1)
            - 'start_date': 적용 시작일 (포함)
            - 'end_date': 적용 종료일 (미포함)
            
    Returns:
        tuple: (engine, results) 백테스트 엔진과 결과 딕셔너리
    """
    engine = None
    results = None
    
    try:
        logger.info("백테스트 엔진 초기화 중...")
        engine = BacktestEngine(
            initial_capital=10000.0,  # 초기 자본: 10,000 USDT
            commission=0.0004,        # 거래 수수료: 0.04%
            leverage=3,               # 레버리지: 3x
            slippage=0.0005,          # 슬리피지: 0.05%
            risk_per_trade=0.02,      # 거래당 리스크: 2%
            force_signal=force_signal  # 강제 신호 전달
        )
        
        logger.info("백테스트 엔진에 데이터 로드 중...")
        engine.load_data(data)
        
        logger.info("트리플 배리어 전략 추가 중...")
        from src.backtesting.strategy.triple_barrier import TripleBarrierStrategy
        engine.add_strategy(TripleBarrierStrategy)
        
        logger.info("성과 분석기 추가 중...")
        engine.add_analyzers()
        
        logger.info("백테스트 실행 중...")
        results = engine.run_backtest()
        
        if results:
            # 결과 요약 출력
            logger.info("\n=== 백테스트 결과 요약 ===")
            logger.info(f"기간: {data['datetime'].min()} ~ {data['datetime'].max()}")
            logger.info(f"초기 자본: {results.get('initial_capital', 0):,.2f} USDT")
            logger.info(f"최종 자산: {results.get('final_value', 0):,.2f} USDT")
            logger.info(f"총 수익률: {results.get('return_pct', 0):.2f}%")
            
            # 주요 성과 지표 출력
            kpis = results.get('kpis', {})
            if kpis:
                logger.info("\n=== 주요 성과 지표 ===")
                logger.info(f"샤프 지수: {kpis.get('sharpe_ratio', 0):.2f}")
                logger.info(f"소르티노 비율: {kpis.get('sortino_ratio', 0):.2f}")
                logger.info(f"최대 낙폭(MDD): {kpis.get('max_drawdown_pct', 0):.2f}%")
                logger.info(f"총 거래 횟수: {kpis.get('number_of_trades', 0)}")
                logger.info(f"승률: {kpis.get('win_rate', 0):.2f}%")
                logger.info(f"수익/손실 비율: {kpis.get('profit_loss_ratio', 0):.2f}")
                logger.info(f"수익 팩터: {kpis.get('profit_factor', 0):.2f}")
                logger.info(f"SQN: {kpis.get('sqn', 0):.2f}")
                logger.info(f"총 거래 비용: {kpis.get('total_commission', 0):.2f} USDT")
            
            # 결과를 파일로 저장
            save_backtest_results(results, kpis)
            
        return engine, results
        
    except Exception as e:
        logger.error(f"백테스트 실행 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def save_backtest_results(results: dict, kpis: dict):
    """백테스트 결과를 파일로 저장합니다."""
    try:
        # 결과 디렉토리 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = REPORT_DIR / f'backtest_{timestamp}'
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 주요 지표 저장
        with open(result_dir / 'summary.txt', 'w', encoding='utf-8') as f:
            f.write("=== 백테스트 결과 요약 ===\n")
            f.write(f"실행 일시: {datetime.now()}\n")
            f.write(f"초기 자본: {results.get('initial_capital', 0):,.2f} USDT\n")
            f.write(f"최종 자산: {results.get('final_value', 0):,.2f} USDT\n")
            f.write(f"총 수익률: {results.get('return_pct', 0):.2f}%\n\n")
            
            f.write("=== 주요 성과 지표 ===\n")
            for key, value in kpis.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # 2. 거래 내역 저장
        if 'trades' in results and results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(result_dir / 'trades.csv', index=False)
        
        # 3. 자산 곡선 시각화
        if 'equity_curve' in results and results['equity_curve']:
            equity_curve = results['equity_curve']
            plt.figure(figsize=(14, 10))
            
            # 자산 곡선
            plt.subplot(2, 1, 1)
            plt.plot(equity_curve['date'], equity_curve['equity'], label='자산 곡선')
            plt.title('자산 곡선')
            plt.xlabel('날짜')
            plt.ylabel('자산 (USDT)')
            plt.grid(True)
            plt.legend()
            
            # 드로다운 곡선
            plt.subplot(2, 1, 2)
            plt.plot(equity_curve['date'], equity_curve['drawdown'], label='드로다운', color='red')
            plt.fill_between(equity_curve['date'], equity_curve['drawdown'], 0, color='red', alpha=0.1)
            plt.title('드로다운 곡선')
            plt.xlabel('날짜')
            plt.ylabel('드로다운 (%)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(result_dir / 'equity_curve.png')
            plt.close()
        
        logger.info(f"백테스트 결과가 저장되었습니다: {result_dir}")
        
    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def plot_equity_curve(engine, save_path=None):
    """백테스트 결과로부터 자산 곡선을 그립니다."""
    if not isinstance(engine, BacktestEngine) or not hasattr(engine, 'results') or not engine.results:
        print("\n경고: 유효한 백테스트 엔진 또는 결과가 없어 자산 곡선을 그릴 수 없습니다.")
        return
        
    try:
        equity_data = engine.results.get('equity_curve')
        
        # equity_curve가 딕셔너리인 경우 pandas Series로 변환
        if isinstance(equity_data, dict):
            if not equity_data:  # 빈 딕셔너리인 경우
                print("경고: 자산 곡선 데이터가 비어 있습니다.")
                return
            # 딕셔너리를 pandas Series로 변환 (날짜 인덱스가 있는 경우)
            equity_curve = pd.Series(equity_data)
            if not isinstance(equity_curve.index, pd.DatetimeIndex):
                # 인덱스가 날짜 형식이 아닌 경우, 단순 정수 인덱스 사용
                equity_curve.index = pd.RangeIndex(len(equity_curve))
        elif hasattr(equity_data, 'empty') and equity_data.empty:
            print("경고: 자산 곡선 데이터가 비어 있습니다.")
            return
        else:
            equity_curve = equity_data

        plt.figure(figsize=(12, 6))
        
        # 시계열 데이터 플로팅 (날짜 인덱스가 있는 경우)
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            plt.plot(equity_curve.index, equity_curve, label='Equity Curve')
            plt.gcf().autofmt_xdate()  # 날짜 레이블 자동 조정
        else:
            # 날짜 인덱스가 없는 경우 단순 인덱스 사용
            plt.plot(equity_curve.values, label='Equity Curve')
            
        plt.title('Backtest Equity Curve')
        plt.xlabel('Date' if isinstance(equity_curve.index, pd.DatetimeIndex) else 'Period')
        plt.ylabel('Equity (USDT)')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            PLOT_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"자산 곡선이 저장되었습니다: {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"\n자산 곡선을 그리는 중 오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()

def main(force_signal: Optional[Dict] = None):
    """
    메인 함수
    
    Args:
        force_signal: 강제로 주입할 신호 (테스트용)
            - 'signal': 고정 신호 값 (-1, 0, 1)
            - 'start_date': 적용 시작일 (포함)
            - 'end_date': 적용 종료일 (미포함)
    """
    try:
        logger.info("=== 백테스트 시작 ===")
        
        # 1. 데이터 로드 및 예측 (강제 신호 전달)
        data = load_data_and_predict(force_signal=force_signal)
        
        # 2. 백테스트 실행 (강제 신호 전달)
        engine, results = run_backtest(data, force_signal=force_signal)
        
        # 3. 결과 시각화
        plot_equity_curve(engine, save_path=PLOT_DIR / 'equity_curve.png')
        
        logger.info("\n=== 백테스트 완료 ===")
        return 0
        
    except Exception as e:
        logger.error(f"백테스트 실행 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    import argparse
    
    # 명령줄 인수 파서 설정
    parser = argparse.ArgumentParser(description='암호화폐 트레이딩 전략 백테스트')
    parser.add_argument('--signal', type=int, choices=[-1, 0, 1], 
                       help='강제로 주입할 신호 값 (-1: 매도, 0: 홀드, 1: 매수)')
    parser.add_argument('--start-date', type=str, 
                       help='강제 신호 적용 시작일 (YYYY-MM-DD 형식)')
    parser.add_argument('--end-date', type=str,
                       help='강제 신호 적용 종료일 (YYYY-MM-DD 형식)')
    
    args = parser.parse_args()
    
    # force_signal 딕셔너리 생성
    force_signal = None
    if args.signal is not None:
        force_signal = {'signal': args.signal}
        if args.start_date:
            force_signal['start_date'] = args.start_date
        if args.end_date:
            force_signal['end_date'] = args.end_date
    
    # 메인 함수 실행
    sys.exit(main(force_signal=force_signal))
