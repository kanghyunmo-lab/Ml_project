"""
퀀텀 리프 프로젝트를 위한 백테스팅 엔진

이 모듈은 모델 예측을 사용한 트레이딩 전략을 백테스트하는 기능을 제공합니다.
현실적인 거래 조건을 시뮬레이션하고 성능 지표를 계산합니다.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any, Union

# Backtrader의 분석기 임포트
try:
    from backtrader.analyzers import SharpeRatio, DrawDown, TradeAnalyzer, Returns, TimeReturn, SQN
except ImportError:
    # 일부 버전에서는 직접 임포트가 안 될 수 있으므로 예외 처리
    SharpeRatio = bt.analyzers.SharpeRatio
    DrawDown = bt.analyzers.DrawDown
    TradeAnalyzer = bt.analyzers.TradeAnalyzer
    Returns = bt.analyzers.Returns
    TimeReturn = bt.analyzers.TimeReturn
    SQN = bt.analyzers.SQN

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """모델 예측을 포함한 트레이딩 전략을 평가하기 위한 백테스팅 엔진"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.0004,
                 leverage: int = 3, slippage: float = 0.0005):
        """백테스팅 엔진을 초기화합니다.
        
        매개변수:
            initial_capital: 초기 자본 (USDT)
            commission: 거래 수수료 비율 (바이낸스 기준 0.04%)
            leverage: 사용할 최대 레버리지 (1-3배)
            slippage: 가격 대비 슬리피지 비율
        """
        self.initial_capital = initial_capital
        self.commission = commission
        # PRD에 따라 최대 레버리지를 3배로 제한
        self.leverage = min(max(leverage, 1), 3)
        self.slippage = slippage
        self.cerebro = bt.Cerebro()  # 백트레이더의 핵심 엔진 초기화
        
        # 백트레이더 설정
        self.cerebro.broker.setcash(initial_capital)  # 초기 자본 설정
        self.cerebro.broker.setcommission(commission=commission)  # 수수료 설정
        self.cerebro.broker.set_slippage_perc(self.slippage)  # 슬리피지 설정
        
        # 결과 저장을 위한 변수 초기화
        self.results = {}
        self.analyzers = {}
    
    def load_data(self, data: pd.DataFrame, price_col: str = 'close', 
                 prediction_col: str = 'prediction', datetime_col: str = 'datetime') -> None:
        """백테스트에 사용할 데이터를 로드합니다.
        
        매개변수:
            data: OHLCV 데이터를 포함한 데이터프레임
            price_col: 가격 컬럼 이름 (기본값: 'close')
            prediction_col: 예측값 컬럼 이름 (기본값: 'prediction')
            datetime_col: 날짜/시간 컬럼 이름 (기본값: 'datetime')
        """
        try:
            # 데이터프레임 복사 (원본 변경 방지)
            df = data.copy()
            
            # datetime 컴럼이 있는 경우에만 처리
            if datetime_col is not None:
                # datetime 컴럼이 문자열인 경우 변환
                if datetime_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                    
                # 인덱스를 datetime으로 설정
                df = df.set_index(datetime_col)
            # datetime_col이 None이면 이미 인덱스가 설정되어 있다고 가정
            
            # 필요한 컬럼이 있는지 확인
            required_cols = ['open', 'high', 'low', 'close', 'volume', prediction_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_cols}")
            
            # 예측값이 범주형인 경우 숫자로 변환 (-1, 0, 1)
            if not pd.api.types.is_numeric_dtype(df[prediction_col]):
                df[prediction_col] = df[prediction_col].map({'sell': -1, 'hold': 0, 'buy': 1})
            
            # Backtrader용 데이터 피드 생성
            class PredictionData(bt.feeds.PandasData):
                lines = ('prediction',)  # 예측값 라인 추가
                params = (('prediction', -1),)
            
            data_feed = PredictionData(
                dataname=df,
                datetime=None,  # 인덱스 사용
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=None,
                prediction=prediction_col  # 모델 예측값
            )
            
            # 데이터 피드를 Cerebro에 추가
            self.cerebro.adddata(data_feed, name='btc_usdt')
            
            # 분석기 추가
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
            
            logger.info(f"Successfully loaded {len(df)} rows of data")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def add_strategy(self, strategy_class, **kwargs):
        """Add a trading strategy to the backtest.
        
        Args:
            strategy_class: The strategy class to use
            **kwargs: Additional arguments to pass to the strategy
        """
        self.cerebro.addstrategy(strategy_class, **kwargs)
    
    def add_analyzers(self):
        """Add performance analyzers."""
        # Add built-in analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')  # For equity curve
    
    def run_backtest(self) -> Dict[str, Any]:
        """백테스트를 실행하고 결과를 반환합니다.
        
        반환값:
            백테스트 결과와 성능 지표를 포함하는 딕셔너리
        """
        try:
            # 분석기 추가
            self.add_analyzers()
            
            # 전략 추가 (아직 추가되지 않은 경우)
            if not self.cerebro.strats:
                self.cerebro.addstrategy(
                    TripleBarrierStrategy,
                    leverage=self.leverage,
                    risk_per_trade=0.02,  # 거래당 위험 비중 (2%)
                    stop_loss_pct=0.01,   # 1% 손절
                    take_profit_pct=0.02   # 2% 익절
                )
            
            # 백테스트 실행
            logger.info("백테스트를 시작합니다...")
            results = self.cerebro.run()
            
            if not results or len(results) == 0:
                raise ValueError("백테스트가 실행되지 않았습니다. 결과가 없습니다.")
            
            # 결과 추출 및 저장
            strategy = results[0]
            self._process_results(strategy)
            
            # 최종 결과 구성
            final_results = {
                'initial_capital': self.initial_capital,
                'final_value': self.results.get('final_value', self.initial_capital),
                'return_pct': self.results.get('return_pct', 0),
                'kpis': self.results.get('kpis', {})
            }
            
            logger.info("백테스트가 성공적으로 완료되었습니다.")
            return final_results
            
        except Exception as e:
            logger.error(f"백테스트 실행 중 오류가 발생했습니다: {str(e)}", exc_info=True)
            raise
    
    def _process_results(self, strategy) -> None:
        """백테스트 결과를 처리하고 저장합니다."""
        try:
            # 기본 메트릭 저장
            final_value = self.cerebro.broker.getvalue()
            return_pct = (final_value / self.initial_capital - 1) * 100
            
            self.results.update({
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'return_pct': return_pct
            })
            
            # 분석기 결과 추출 (에러 방지를 위한 안전한 접근)
            analyzers = {}
            try:
                analyzers['sharpe'] = strategy.analyzers.sharpe.get_analysis()
            except Exception as e:
                logger.warning(f"샤프 지수 계산 중 오류: {str(e)}")
                analyzers['sharpe'] = {'sharperatio': 0.0}
                
            try:
                analyzers['drawdown'] = strategy.analyzers.drawdown.get_analysis()
            except Exception as e:
                logger.warning(f"드로다운 계산 중 오류: {str(e)}")
                analyzers['drawdown'] = {'max': {'drawdown': 0.0}}
                
            try:
                analyzers['trades'] = strategy.analyzers.trades.get_analysis()
            except Exception as e:
                logger.warning(f"거래 분석 중 오류: {str(e)}")
                analyzers['trades'] = {'total': {'total': 0, 'open': 0, 'closed': 0}, 
                                     'won': {'total': 0, 'pnl': {'total': 0}}, 
                                     'lost': {'total': 0, 'pnl': {'total': 0}}}
            
            self.analyzers = analyzers
            
            # KPI 계산
            self._calculate_kpis()
            
            # 에퀴티 곡선 계산
            self._calculate_equity_curve()
            
        except Exception as e:
            logger.error(f"결과 처리 중 오류: {str(e)}", exc_info=True)
            raise
    
    def _calculate_kpis(self) -> None:
        """주요 성과 지표를 계산하고 저장합니다."""
        try:
            # 분석기에서 메트릭 안전하게 추출
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            
            if 'sharpe' in self.analyzers:
                sharpe_analysis = self.analyzers['sharpe'].get_analysis()
                sharpe_ratio = float(sharpe_analysis.get('sharperatio', 0.0))
            
            if 'drawdown' in self.analyzers:
                drawdown_analysis = self.analyzers['drawdown'].get_analysis()
                max_drawdown = float(drawdown_analysis.get('max', {}).get('drawdown', 0.0))
            
            # 거래 분석 결과 추출
            total_trades = 0
            total_wins = 0
            total_losses = 0
            gross_wins = 0.0
            gross_losses = 0.0
            
            if 'trades' in self.analyzers:
                trades_analysis = self.analyzers['trades'].get_analysis()
                total_trades = int(trades_analysis.get('total', {}).get('closed', 0))
                total_wins = int(trades_analysis.get('won', {}).get('total', 0))
                total_losses = int(trades_analysis.get('lost', {}).get('total', 0))
                
                # 수익/손실 금액 (양수/음수)
                gross_wins = float(trades_analysis.get('won', {}).get('pnl', {}).get('total', 0.0) or 0.0)
                gross_losses = abs(float(trades_analysis.get('lost', {}).get('pnl', {}).get('total', 0.0) or 0.0))
            
            # 승률 계산
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
            
            # Profit Factor (총 수익 / 총 손실)
            profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float('inf')
            
            # KPI 딕셔너리 생성
            kpis = {
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown_pct': round(max_drawdown, 2),
                'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
                'win_rate': round(win_rate, 2),
                'number_of_trades': total_trades,
                'win_trades': total_wins,
                'loss_trades': total_losses,
                'total_return_pct': round(self.results.get('return_pct', 0.0), 2)
            }
            
            # 결과에 KPI 저장
            self.results['kpis'] = kpis
            
        except Exception as e:
            logger.error(f"KPI 계산 중 오류: {str(e)}", exc_info=True)
            # 오류 발생 시 기본값으로 채운 KPI 반환
            self.results['kpis'] = {
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'number_of_trades': 0,
                'win_trades': 0,
                'loss_trades': 0,
                'total_return_pct': 0.0
            }
    
    def _calculate_equity_curve(self) -> None:
        """에퀴티 곡선을 계산하고 저장합니다."""
        try:
            # TimeReturn 분석기에서 수익률 시계열 가져오기
            if hasattr(self, 'analyzers') and 'timereturn' in self.analyzers:
                timereturn = self.analyzers['timereturn']
                if hasattr(timereturn, 'get_analysis'):
                    returns = timereturn.get_analysis()
                    if returns:
                        self.equity_curve = pd.Series(returns)
                        # 누적 수익률로 변환
                        self.equity_curve = (1 + self.equity_curve).cumprod() - 1
                        # 초기 자본을 곱하여 실제 가치로 변환
                        self.equity_curve = (1 + self.equity_curve) * self.initial_capital
                        # 결과에 저장
                        self.results['equity_curve'] = self.equity_curve.to_dict()
                        return
            
            # 분석기에서 가져오지 못한 경우 빈 시리즈 생성
            self.equity_curve = pd.Series()
            self.results['equity_curve'] = {}
            
        except Exception as e:
            logger.error(f"에퀴티 곡선 계산 중 오류: {str(e)}", exc_info=True)
            self.equity_curve = pd.Series()
            self.results['equity_curve'] = {}
    
    def get_summary(self) -> Dict[str, Any]:
        """백테스트 결과 요약을 반환합니다."""
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.results.get('final_value', self.initial_capital),
            'return_pct': self.results.get('return_pct', 0),
            'kpis': self.results.get('kpis', {})
        }
    
    def generate_report(self, output_dir: str = 'reports') -> str:
        """상세 백테스트 보고서를 생성합니다.
        
        매개변수:
            output_dir: 보고서를 저장할 디렉토리 경로 (기본값: 'reports')
            
        반환값:
            생성된 보고서 파일 경로
        """
        try:
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join(output_dir, f'backtest_report_{timestamp}.json')
            
            # 보고서 데이터 준비
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'initial_capital': self.initial_capital,
                    'commission': self.commission,
                    'leverage': self.leverage,
                    'slippage': self.slippage
                },
                'results': self.results,
                'kpis': self.results.get('kpis', {})
            }
            
            # 분석기 결과가 있는 경우에만 추가
            if hasattr(self, 'analyzers'):
                analyzers_data = {}
                for name, analyzer in self.analyzers.items():
                    try:
                        analyzers_data[name] = analyzer.get_analysis()
                    except Exception as e:
                        logger.warning(f"분석기 {name}에서 데이터를 가져오는 중 오류: {str(e)}")
                        analyzers_data[name] = str(e)
                
                if analyzers_data:
                    report_data['analyzers'] = analyzers_data
            
            # 보고서 저장
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"백테스트 보고서가 저장되었습니다: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류가 발생했습니다: {str(e)}", exc_info=True)
            raise


class TripleBarrierStrategy(bt.Strategy):
    """삼중 장벽 기법 예측을 기반으로 한 트레이딩 전략"""
    
    # 전략 파라미터 설정
    params = (
        ('leverage', 3),  # PRD에 따른 최대 레버리지
        ('risk_per_trade', 0.02),  # 거래당 2% 리스크
        ('stop_loss_pct', 0.01),  # 1% 스탑로스
        ('take_profit_pct', 0.02),  # 2% 익절
        ('max_position_size', 0.1),  # 최대 포지션 크기 (전체 자본 대비)
    )
    
    def __init__(self):
        """전략을 초기화합니다."""
        # 예측 데이터 가져오기 (데이터 피드에 prediction 라인이 있어야 함)
        self.prediction = self.datas[0].prediction
        self.order = None  # 주문 객체 초기화
        self.trade_count = 0  # 거래 횟수 카운터
        self.entry_price = 0  # 진입 가격 추적
        self.stop_loss = 0  # 스탑로스 가격
        self.take_profit = 0  # 익절 가격
    
    def next(self):
        """Process each bar."""
        if self.order:
            return  # 보류 중인 주문이 있으면 반환
            
        # 현재 예측값 가져오기
        current_pred = int(self.prediction[0])  # -1, 0, 1 중 하나여야 함
        current_price = self.data.close[0]
        
        # 포지션이 없을 때 진입 신호 확인
        if not self.position:
            if current_pred == 1:  # 매수 신호
                self.enter_long(current_price)
            elif current_pred == -1:  # 매도 신호
                self.enter_short(current_price)
        else:
            # 포지션이 있을 때 청산 신호 확인
            if (self.position.size > 0 and current_pred == -1) or \
               (self.position.size < 0 and current_pred == 1):
                self.close()
            # 스탑로스/익절 확인
            elif self.position.size > 0:  # 롱 포지션
                if current_price <= self.stop_loss or current_price >= self.take_profit:
                    self.close()
            else:  # 숏 포지션
                if current_price >= self.stop_loss or current_price <= self.take_profit:
                    self.close()
    
    def enter_long(self, current_price):
        """롱 포지션을 진입합니다. 리스크 관리가 적용됩니다."""
        # 스탑로스와 익절 가격 계산
        stop_loss = current_price * (1 - self.p.stop_loss_pct)  # 예: 1% 스탑로스
        take_profit = current_price * (1 + self.p.take_profit_pct)  # 예: 2% 익절
        
        # 포지션 사이즈 계산 (리스크 기반)
        risk_amount = self.broker.getvalue() * self.p.risk_per_trade
        price_diff = current_price - stop_loss
        size = (risk_amount / price_diff) * self.p.leverage
        
        # 최대 포지션 크기 제한
        max_size = (self.broker.getvalue() * self.p.max_position_size) / current_price
        size = min(size, max_size)
        
        if size > 0:
            # 주문 실행
            self.order = self.buy(size=size)
            self.entry_price = current_price
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.trade_count += 1
            logger.info(f"LONG ENTRY - Price: {current_price:.2f}, "
                      f"Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}, "
                      f"Size: {size:.4f}")
    
    def enter_short(self, current_price):
        """숏 포지션을 진입합니다. 리스크 관리가 적용됩니다."""
        # 스탑로스와 익절 가격 계산
        stop_loss = current_price * (1 + self.p.stop_loss_pct)  # 예: 1% 스탑로스
        take_profit = current_price * (1 - self.p.take_profit_pct)  # 예: 2% 익절
        
        # 포지션 사이즈 계산 (리스크 기반)
        risk_amount = self.broker.getvalue() * self.p.risk_per_trade
        price_diff = stop_loss - current_price
        size = (risk_amount / price_diff) * self.p.leverage
        
        # 최대 포지션 크기 제한
        max_size = (self.broker.getvalue() * self.p.max_position_size) / current_price
        size = min(size, max_size)
        
        if size > 0:
            # 주문 실행
            self.order = self.sell(size=size)
            self.entry_price = current_price
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.trade_count += 1
            logger.info(f"SHORT ENTRY - Price: {current_price:.2f}, "
                      f"Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}, "
                      f"Size: {size:.4f}")
    
    def notify_order(self, order):
        """주문 알림을 처리합니다."""
        if order.status in [order.Submitted, order.Accepted]:
            return  # 실행 대기 중
            
        if order.status in [order.Completed]:
            if order.isbuy():
                log_text = (
                    f"BUY EXECUTED - Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
                # 롱 포지션 진입 시 포지션 정보 업데이트
                if not hasattr(self, 'position_size'):
                    self.position_size = order.executed.size
                    self.position_value = order.executed.value
                    self.entry_price = order.executed.price
                    logger.info(f"LONG POSITION OPENED - Size: {self.position_size:.4f}, "
                              f"Entry: {self.entry_price:.2f}")
                
            elif order.issell():
                log_text = (
                    f"SELL EXECUTED - Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
                # 숏 포지션 진입 시 포지션 정보 업데이트
                if not hasattr(self, 'position_size') or self.position_size == 0:
                    self.position_size = -order.executed.size  # 음수로 표시
                    self.position_value = order.executed.value
                    self.entry_price = order.executed.price
                    logger.info(f"SHORT POSITION OPENED - Size: {abs(self.position_size):.4f}, "
                              f"Entry: {self.entry_price:.2f}")
                # 포지션 청산 시
                elif (self.position_size > 0 and order.executed.size > 0) or \
                     (self.position_size < 0 and order.executed.size < 0):
                    pnl = (order.executed.price - self.entry_price) * order.executed.size
                    pnl_pct = (pnl / (abs(self.entry_price * order.executed.size))) * 100
                    logger.info(f"POSITION CLOSED - PnL: {pnl:.2f} ({pnl_pct:.2f}%)")
                    self.position_size = 0
                    self.position_value = 0
            
            logger.info(log_text)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order {order.getstatusname()}: {order.info if hasattr(order, 'info') else ''}")
        
        self.order = None  # 주문 객체 초기화
    
    def stop(self):
        """백테스트 종료 시 호출됩니다."""
        # 최종 포트폴리오 가치 및 수익률 계산
        final_value = self.broker.getvalue()
        pnl = final_value - self.broker.startingcash
        pnl_pct = (pnl / self.broker.startingcash) * 100
        
        # 거래 통계 로깅
        logger.info("=" * 70)
        logger.info(f"{'BACKTEST COMPLETED':^70}")
        logger.info("=" * 70)
        logger.info(f"{'Initial Portfolio Value:':<30} {self.broker.startingcash:>20.2f} USDT")
        logger.info(f"{'Final Portfolio Value:':<30} {final_value:>20.2f} USDT")
        logger.info(f"{'Profit/Loss:':<30} {pnl:>+20.2f} USDT ({pnl_pct:+.2f}%)")
        logger.info(f"{'Total Trades:':<30} {self.trade_count:>20}")
        
        # 분석기 결과가 있으면 추가 정보 표시
        if hasattr(self, 'analyzers'):
            if hasattr(self.analyzers.sharpe, 'get_analysis'):
                sharpe = self.analyzers.sharpe.get_analysis()
                if 'sharperatio' in sharpe:
                    logger.info(f"{'Sharpe Ratio:':<30} {sharpe['sharperatio']:>20.2f}")
                    
            if hasattr(self.analyzers.drawdown, 'get_analysis'):
                drawdown = self.analyzers.drawdown.get_analysis()
                if 'max' in drawdown and 'drawdown' in drawdown['max']:
                    logger.info(f"{'Max Drawdown:':<30} {drawdown['max']['drawdown']:>19.2f}%")
        
        logger.info("=" * 70)
