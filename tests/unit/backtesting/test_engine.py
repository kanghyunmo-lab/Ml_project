"""Unit tests for the backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.backtesting.engine import BacktestEngine, TripleBarrierStrategy

# Sample test data
def create_sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='4H')
    np.random.seed(42)
    
    # Create sample data with some trend
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close + np.abs(np.random.randn(100))
    low = close - np.abs(np.random.randn(100))
    open_price = np.roll(close, 1)
    open_price[0] = 100  # First open
    
    # Add predictions (-1, 0, 1)
    predictions = np.random.choice([-1, 0, 1], size=100, p=[0.3, 0.4, 0.3])
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100, 1000, 100),
        'prediction': predictions
    })
    
    return df

class TestBacktestEngine:
    """Test cases for the BacktestEngine class."""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data file for testing."""
        df = create_sample_data()
        file_path = tmp_path / 'test_data.parquet'
        df.to_parquet(file_path)
        return file_path
    
    def test_initialization(self):
        """Test BacktestEngine initialization."""
        engine = BacktestEngine(
            initial_capital=10000.0,
            commission=0.0004,
            leverage=3,
            slippage=0.0005
        )
        
        assert engine.initial_capital == 10000.0
        assert engine.commission == 0.0004
        assert engine.leverage == 3
        assert engine.slippage == 0.0005
    
    def test_load_data(self, sample_data):
        """Test loading data into the backtest engine."""
        engine = BacktestEngine()
        engine.load_data(str(sample_data))
        
        # Check if data is loaded into cerebro
        assert len(engine.cerebro.datas) > 0
    
    def test_add_strategy(self):
        """Test adding a strategy to the backtest engine."""
        engine = BacktestEngine()
        engine.add_strategy(TripleBarrierStrategy)
        
        # Check if strategy is added
        assert len(engine.cerebro.strats) > 0
    
    def test_run_backtest(self, sample_data):
        """Test running a backtest with sample data."""
        engine = BacktestEngine(initial_capital=10000.0)
        engine.load_data(str(sample_data))
        engine.add_strategy(TripleBarrierStrategy)
        
        # Run backtest
        results = engine.run_backtest()
        
        # Check if results are returned
        assert isinstance(results, dict)
        assert 'final_value' in results
        assert 'kpis' in results
        
        # Check if KPIs are calculated
        kpis = results['kpis']
        assert 'sharpe_ratio' in kpis
        assert 'max_drawdown_pct' in kpis
        assert 'total_return_pct' in kpis
        assert 'calmar_ratio' in kpis
        assert 'profit_factor' in kpis
        assert 'win_rate' in kpis
        assert 'sqn' in kpis
        assert 'number_of_trades' in kpis
    
    def test_generate_report(self, sample_data, tmp_path):
        """Test generating a backtest report."""
        engine = BacktestEngine(initial_capital=10000.0)
        engine.load_data(str(sample_data))
        engine.add_strategy(TripleBarrierStrategy)
        
        # Run backtest and generate report
        engine.run_backtest()
        report_path = engine.generate_report(output_dir=str(tmp_path))
        
        # Check if report file is created
        assert os.path.exists(report_path)
        assert report_path.endswith('.json')


class TestTripleBarrierStrategy:
    """Test cases for the TripleBarrierStrategy class."""
    
    def test_enter_long(self):
        """Test entering a long position."""
        # This would require more complex setup with mock data
        pass
    
    def test_enter_short(self):
        """Test entering a short position."""
        # This would require more complex setup with mock data
        pass


if __name__ == "__main__":
    pytest.main(["-v", "--cov=src.backtesting.engine", "--cov-report=term-missing"])
    # To run with coverage: python -m pytest tests/unit/backtesting/test_engine.py -v --cov=src.backtesting.engine --cov-report=term-missing
