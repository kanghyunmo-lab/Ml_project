"""
전체 트레이딩 파이프라인을 실행하는 예제 스크립트
"""
import os
import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# 필요한 모듈 임포트
import sys

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 필요한 모듈 임포트
from src.collection.binance_collector import BinanceDataCollector
from src.processing.audit import run_data_audit
from src.processing.clean import clean_ohlcv_data
from src.processing.feature_engineer import generate_features
from src.modeling.predictor import load_model, predict
from src.backtesting.engine import BacktestEngine
from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.utils.helper import create_directories

# 필요한 디렉토리 생성
create_directories([DATA_DIR, MODELS_DIR, RESULTS_DIR])

HAS_DEPENDENCIES = True

def run_pipeline(symbol="BTC/USDT", timeframe="4h", days=30, initial_capital=10000.0, leverage=3.0):
    """
    전체 트레이딩 파이프라인을 실행하는 함수
    
    Args:
        symbol (str): 거래할 암호화폐 심볼 (예: "BTC/USDT")
        timeframe (str): 시간 단위 (예: "4h", "1d")
        days (int): 가져올 데이터 일수
        initial_capital (float): 초기 자본 (USDT)
        leverage (float): 사용할 레버리지 배수
    """
    if not HAS_DEPENDENCIES:
        logger.error("필요한 의존성이 설치되지 않아 파이프라인을 실행할 수 없습니다.")
        return
    
    logger.info(f"{symbol}에 대한 트레이딩 파이프라인을 시작합니다...")
    
    # 1. 데이터 수집
    logger.info("1. 데이터 수집 중...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    raw_data_path = DATA_DIR / f"{symbol.lower().replace('/', '_')}_{timeframe}_raw.parquet"
    
    try:
        # BinanceDataCollector 인스턴스 생성
        collector = BinanceDataCollector(base_dir=DATA_DIR)
        
        # 데이터 수집 (start_date과 end_date을 문자열로 변환하여 전달)
        df = collector.collect_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            save_path=raw_data_path
        )
        logger.info(f"   - {len(df)}개의 캔들 데이터 수집 완료")
    except Exception as e:
        logger.error(f"데이터 수집 중 오류가 발생했습니다: {e}")
        return
    
    # 2. 데이터 품질 감사
    logger.info("2. 데이터 품질 감사 중...")
    try:
        audit_report = run_data_audit(df, symbol=symbol)
        logger.info(f"   - 감사 완료: {audit_report['missing_values']}개의 결측치 발견")
    except Exception as e:
        logger.error(f"데이터 감사 중 오류가 발생했습니다: {e}")
        return
    
    # 3. 데이터 정제
    logger.info("3. 데이터 정제 중...")
    processed_data_path = DATA_DIR / f"{symbol.lower().replace('/', '_')}_{timeframe}_processed.parquet"
    
    try:
        cleaned_df = clean_ohlcv_data(df, save_path=processed_data_path)
        logger.info(f"   - {len(cleaned_df)}개의 캔들 데이터 정제 완료")
    except Exception as e:
        logger.error(f"데이터 정제 중 오류가 발생했습니다: {e}")
        return
    
    # 4. 피처 엔지니어링
    logger.info("4. 피처 엔지니어링 중...")
    features_path = DATA_DIR / f"{symbol.lower().replace('/', '_')}_{timeframe}_features.parquet"
    
    try:
        features_df = generate_features(cleaned_df, save_path=features_path)
        logger.info(f"   - {len(features_df.columns) - len(cleaned_df.columns)}개의 피처 생성 완료")
    except Exception as e:
        logger.error(f"피처 엔지니어링 중 오류가 발생했습니다: {e}")
        return
    
    # 5. 모델 로드 및 예측
    logger.info("5. 모델 로드 및 예측 중...")
    model_path = MODELS_DIR / "lgbm_model.pkl"
    
    try:
        if model_path.exists():
            # 모델이 존재하는 경우 로드
            logger.info(f"기존 모델을 로드합니다: {model_path}")
            model = load_model(model_path)
        else:
            # 모델이 없는 경우, 간단한 모델을 훈련시켜 사용
            logger.info(f"모델을 찾을 수 없어 새로운 모델을 훈련합니다.")
            
            # 훈련 데이터 준비
            train_df = features_df.copy()
            
            # 다음 캔들의 종가로 타겟 생성 (상승:1, 하락:0)
            train_df['next_close'] = train_df['close'].shift(-1)
            train_df['target'] = (train_df['next_close'] > train_df['close']).astype(int)
            train_df = train_df.dropna()  # NaN 제거
            
            # 특성 및 타겟 분리
            feature_cols = [col for col in train_df.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume', 
                                      'datetime', 'timestamp', 'next_close', 'target']]
            
            X = train_df[feature_cols]
            y = train_df['target']
            
            # 가장 기본적인 LightGBM 모델 훈련
            try:
                import lightgbm as lgbm
                from sklearn.model_selection import train_test_split
                
                # 훈련/검증 분할
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
                
                # 모델 파라미터
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'learning_rate': 0.05,
                    'verbose': -1
                }
                
                # 모델 훈련
                train_data = lgbm.Dataset(X_train, label=y_train)
                valid_data = lgbm.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgbm.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[valid_data],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
                
                # 모델 저장
                import joblib
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path)
                logger.info(f"새 모델이 훈련되어 저장되었습니다: {model_path}")
                
            except ImportError as e:
                logger.error(f"LightGBM 또는 관련 패키지를 가져올 수 없습니다: {e}")
                logger.info("랜덤 신호를 사용하여 계속합니다.")
                
                # LightGBM이 없는 경우 랜덤 예측 사용
                import numpy as np
                predictions = np.random.choice([-1, 0, 1], size=len(features_df), p=[0.3, 0.4, 0.3])
                features_df['signal'] = predictions
                logger.info(f"   - {len(predictions)}개의 랜덤 신호 생성 완료")
                return
        
        # 예측 수행
        predictions = predict(model, features_df)
        features_df['signal'] = predictions
        logger.info(f"   - {len(predictions)}개의 예측 완료")
        
    except Exception as e:
        logger.error(f"예측 중 오류가 발생했습니다: {e}")
        # 오류 발생 시 랜덤 신호로 대체
        import numpy as np
        predictions = np.random.choice([-1, 0, 1], size=len(features_df), p=[0.3, 0.4, 0.3])
        features_df['signal'] = predictions
        logger.info(f"   - 오류로 인해 {len(predictions)}개의 랜덤 신호 생성 완료")
    
    # 6. 백테스트 실행
    logger.info("6. 백테스트 실행 중...")
    
    try:
        # 백테스트 엔진 초기화
        engine = BacktestEngine(
            data=features_df,
            initial_capital=initial_capital,
            commission=0.0004,  # 0.04% 수수료
            leverage=leverage,
            risk_per_trade=0.02  # 거래당 2% 리스크
        )
        
        # 백테스트 실행
        results = engine.run()
        
        # 결과 출력
        logger.info("\n=== 백테스트 결과 ===")
        logger.info(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"초기 자본: {initial_capital:,.2f} USDT")
        logger.info(f"최종 자본: {results.get('final_balance', initial_capital):,.2f} USDT")
        logger.info(f"총 수익률: {results.get('total_return', 0):.2%}")
        logger.info(f"최대 낙폭: {results.get('max_drawdown', 0):.2%}")
        logger.info(f"샤프 지수: {results.get('sharpe_ratio', 'N/A'):.2f}" if 'sharpe_ratio' in results else "샤프 지수: N/A")
        logger.info(f"총 거래 횟수: {results.get('total_trades', 0)}회")
        logger.info(f"승률: {results.get('win_rate', 0):.2%}")
        logger.info("==================")
        
        # 결과 저장
        result_df = pd.DataFrame([{
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date.strftime("%Y-%m-%d"),
            'end_date': end_date.strftime("%Y-%m-%d"),
            'initial_capital': initial_capital,
            'final_balance': results.get('final_balance', initial_capital),
            'total_return': results.get('total_return', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'sharpe_ratio': results.get('sharpe_ratio', None),
            'total_trades': results.get('total_trades', 0),
            'win_rate': results.get('win_rate', 0),
            'leverage': leverage,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        
        # 결과를 CSV 파일로 저장
        result_file = RESULTS_DIR / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df.to_csv(result_file, index=False)
        logger.info(f"백테스트 결과가 저장되었습니다: {result_file}")
        
    except Exception as e:
        logger.error(f"백테스트 실행 중 오류가 발생했습니다: {e}")
        return
    
    logger.info("파이프라인 실행이 완료되었습니다.")

if __name__ == "__main__":
    # 기본 파라미터로 파이프라인 실행
    run_pipeline(
        symbol="BTC/USDT",
        timeframe="4h",
        days=30,  # 최근 30일 데이터 사용
        initial_capital=10000.0,  # 초기 자본 10,000 USDT
        leverage=3.0  # 3배 레버리지
    )
