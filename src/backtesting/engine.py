"""
퀀텀 리프 프로젝트를 위한 백테스팅 엔진

이 모듈은 모델 예측을 사용한 트레이딩 전략을 백테스트하는 기능을 제공합니다.
현실적인 거래 조건을 시뮬레이션하고 성능 지표를 계산합니다.
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Optional, List, Any
import json
import os

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
    
    def load_data(self, data_path: str) -> None:
        """OHLCV 데이터를 백트레이더에 로드합니다.
        
        매개변수:
            data_path: OHLCV 데이터 파일 경로 (CSV 또는 Parquet 형식)
        """
        try:
            # Load data based on file extension
            if data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:  # Assume CSV
                df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
            
            # Ensure proper column names
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Create a custom data feed class that includes the prediction field
            class PredictionData(bt.feeds.PandasData):
                # Add the 'prediction' line to the lines of the data feed
                lines = ('prediction',)
                
                # Define the parameters for the prediction line
                params = (
                    ('prediction', -1),  # Default value if no prediction column exists
                )
            
            # Create the data feed with the prediction field
            data = PredictionData(
                dataname=df,
                datetime=None,  # Use index
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume',
                openinterest=None,
                prediction='prediction'  # Map the prediction column
            )
            
            self.cerebro.adddata(data)
            logger.info(f"Successfully loaded data from {data_path}")
            
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
        """Run the backtest and return results.
        
        Returns:
            Dictionary containing backtest results and performance metrics
        """
        try:
            # Add analyzers
            self.add_analyzers()
            
            # Run the backtest
            logger.info("Starting backtest...")
            results = self.cerebro.run()
            
            # Extract and store results
            self._process_results(results[0])
            
            logger.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            raise
    
    def _process_results(self, strategy) -> None:
        """Process and store backtest results."""
        # Store basic metrics
        self.results['initial_capital'] = self.initial_capital
        self.results['final_value'] = self.cerebro.broker.getvalue()
        self.results['return_pct'] = (self.results['final_value'] / self.initial_capital - 1) * 100
        
        # Extract analyzer results
        self.analyzers['sharpe'] = strategy.analyzers.sharpe.get_analysis()
        self.analyzers['drawdown'] = strategy.analyzers.drawdown.get_analysis()
        self.analyzers['returns'] = strategy.analyzers.returns.get_analysis()
        self.analyzers['trades'] = strategy.analyzers.trades.get_analysis()
        self.analyzers['sqn'] = strategy.analyzers.sqn.get_analysis()
        self.analyzers['timereturn'] = strategy.analyzers.timereturn.get_analysis()
        
        # Calculate equity curve and other KPIs
        self._calculate_equity_curve()
        self._calculate_kpis()
    
    def _calculate_kpis(self) -> None:
        """Calculate and store key performance indicators."""
        # Safely extract metrics from analyzers using .get() with defaults
        sharpe_ratio = self.analyzers.get('sharpe', {}).get('sharperatio') or 0
        max_drawdown = self.analyzers.get('drawdown', {}).get('max', {}).get('drawdown', 0)
        total_return = self.analyzers.get('returns', {}).get('rtot', 0) * 100  # as percentage

        # Safely access trade analyzer results
        trades_analysis = self.analyzers.get('trades', {})
        total_trades = trades_analysis.get('total', {}).get('closed', 0)
        
        won_analysis = trades_analysis.get('won', {})
        total_wins = won_analysis.get('total', 0)
        gross_wins = won_analysis.get('pnl', {}).get('total', 0)

        lost_analysis = trades_analysis.get('lost', {})
        gross_losses = lost_analysis.get('pnl', {}).get('total', 0)  # This is a negative number

        # Calculate Calmar Ratio (Return/MaxDD)
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        # Calculate Profit Factor
        profit_factor = gross_wins / abs(gross_losses) if gross_losses != 0 else float('inf')

        # Store KPIs
        self.results['kpis'] = {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_return_pct': total_return,
            'calmar_ratio': calmar_ratio,
            'profit_factor': profit_factor,
            'win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
            'sqn': self.analyzers.get('sqn', {}).get('sqn', 0),
            'number_of_trades': total_trades
        }
    
    def _calculate_equity_curve(self) -> None:
        """Calculate and store the equity curve from TimeReturn analyzer."""
        returns = pd.Series(self.analyzers.get('timereturn', {}))
        if not returns.empty:
            cumulative_returns = (1 + returns).cumprod()
            equity_curve = self.initial_capital * cumulative_returns
            equity_curve.index = pd.to_datetime(equity_curve.index)
            self.results['equity_curve'] = equity_curve.to_frame('equity')
        else:
            self.results['equity_curve'] = pd.DataFrame(columns=['equity'])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the backtest results."""
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.results['final_value'],
            'total_return_pct': self.results['return_pct'],
            **self.results.get('kpis', {})
        }
    
    def generate_report(self, output_dir: str = 'reports') -> str:
        """Generate a detailed backtest report.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(output_dir, f'backtest_report_{timestamp}.json')
        
        # Prepare report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'leverage': self.leverage,
                'slippage': self.slippage
            },
            'results': self.results,
            'analyzers': {
                'sharpe': self.analyzers['sharpe'].get_analysis(),
                'drawdown': self.analyzers['drawdown'].get_analysis(),
                'returns': self.analyzers['returns'].get_analysis(),
                'trades': self.analyzers['trades'].get_analysis(),
                'sqn': self.analyzers['sqn'].get_analysis()
            },
            'kpis': self.results.get('kpis', {})
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_path, f, indent=2, default=str)
        
        logger.info(f"Backtest report saved to {report_path}")
        return report_path


class TripleBarrierStrategy(bt.Strategy):
    """삼중 장벽 기법 예측을 기반으로 한 트레이딩 전략"""
    
    # 전략 파라미터 설정
    params = (
        ('leverage', 3),  # PRD에 따른 최대 레버리지
        ('risk_per_trade', 0.02),  # 거래당 2% 리스크
    )
    
    def __init__(self):
        """전략을 초기화합니다."""
        self.data_pred = self.datas[0].prediction  # 예측 데이터
        self.order = None  # 주문 객체 초기화
        self.trade_count = 0  # 거래 횟수 카운터
    
    def next(self):
        """Process each bar."""
        if self.order:
            return  # Pending order exists
            
        # Get current prediction
        prediction = self.data_pred[0]
        
        # No position - check for entry
        if not self.position:
            if prediction == 1:  # Buy signal
                self.enter_long()
            elif prediction == -1:  # Short signal (if enabled)
                self.enter_short()
        else:
            # Check for exit based on prediction
            if (self.position.size > 0 and prediction == -1) or \
               (self.position.size < 0 and prediction == 1):
                self.close()
    
    def enter_long(self):
        """Enter a long position with proper risk management."""
        # Calculate position size based on 2% risk
        stop_loss = self.data.close[0] * 0.99  # Example: 1% stop loss
        risk_amount = self.broker.getvalue() * self.p.risk_per_trade
        price_diff = self.data.close[0] - stop_loss
        size = (risk_amount / price_diff) * self.p.leverage
        
        # Place order
        self.order = self.buy(size=size)
        self.trade_count += 1
    
    def enter_short(self):
        """Enter a short position with proper risk management."""
        # Similar to enter_long but for short positions
        stop_loss = self.data.close[0] * 1.01  # Example: 1% stop loss
        risk_amount = self.broker.getvalue() * self.p.risk_per_trade
        price_diff = stop_loss - self.data.close[0]
        size = (risk_amount / price_diff) * self.p.leverage
        
        # Place order
        self.order = self.sell(size=size)
        self.trade_count += 1
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return  # Awaiting execution
            
        if order.status in [order.Completed]:
            if order.isbuy():
                log_text = (
                    f"BUY EXECUTED - Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
            elif order.issell():
                log_text = (
                    f"SELL EXECUTED - Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
            
            logger.info(log_text)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order Canceled/Margin/Rejected: {order.getstatusname()}")
        
        self.order = None  # Reset order
    
    def stop(self):
        """Called once at the end of the backtest."""
        logger.info(f"Final Portfolio Value: {self.broker.getvalue():.2f}")
        logger.info(f"Total Trades: {self.trade_count}")
