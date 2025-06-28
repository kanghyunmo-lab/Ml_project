"""
전체 트레이딩 파이프라인을 실행하는 예제 스크립트
"""
import os
import sys
import logging
import traceback
import pandas as pd
import numpy as np
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
            # 모델이 없는 경우, 새로운 모델을 훈련
            logger.info("새로운 모델을 훈련합니다...")
            
            # 훈련 데이터 준비
            train_df = features_df.copy()
            
            # 다음 캔들의 종가로 타겟 생성 (상승:1, 하락:0)
            train_df['next_close'] = train_df['close'].shift(-1)
            train_df['target'] = (train_df['next_close'] > train_df['close']).astype(int)
            train_df = train_df.dropna()  # NaN 제거
            
            if len(train_df) < 100:  # 최소 100개 이상의 샘플 필요
                raise ValueError(f"훈련 데이터가 부족합니다. {len(train_df)}개 샘플만 사용 가능 (최소 100개 필요)")
            
            # 특성 및 타겟 분리
            feature_cols = [col for col in train_df.columns 
                         if col not in ['open', 'high', 'low', 'close', 'volume', 
                                     'datetime', 'timestamp', 'next_close', 'target']]
            
            X = train_df[feature_cols]
            y = train_df['target']
            
            try:
                import lightgbm as lgb
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, classification_report
                
                logger.info("새로운 모델을 훈련합니다...")
                
                # 라벨 생성 (다음 캔들의 가격 방향: 상승=1, 하락=0)
                features_df['target'] = (features_df['close'].shift(-1) > features_df['close']).astype(int)
                
                # 마지막 행(타겟이 없는) 제거
                features_df = features_df.dropna(subset=['target'])
                
                # 특성과 타겟 분리
                feature_cols = [col for col in features_df.columns if col not in ['target', 'signal', 'prediction', 'pred_proba']
                              and not col.startswith(('close_', 'volume_', 'high_', 'low_', 'open_'))]  # 원본 OHLCV 제외
                
                # 수치형 특성만 선택
                numeric_cols = features_df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
                X = features_df[numeric_cols]
                y = features_df['target']
                
                # 학습/검증 세트 분할 (80/20)
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
                
                # LightGBM 모델 훈련
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'seed': 42
                }
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    num_boost_round=1000,
                    early_stopping_rounds=50,
                    verbose_eval=10
                )
                
                # 검증 세트로 성능 평가
                val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
                val_pred = (val_pred_proba > 0.5).astype(int)
                
                # 성능 메트릭 계산
                accuracy = accuracy_score(y_val, val_pred)
                logger.info(f"검증 세트 정확도: {accuracy:.4f}")
                logger.info("\n" + classification_report(y_val, val_pred))
                
                # 모델 저장
                import joblib
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path)
                logger.info(f"모델이 성공적으로 훈련되고 저장되었습니다: {model_path}")
                
            except ImportError as e:
                logger.error("LightGBM이 설치되어 있지 않아 모델 훈련을 수행할 수 없습니다.")
                logger.error("pip install lightgbm 명령으로 설치해 주세요.")
                raise
            
        # 예측 수행
        try:
            logger.info("예측을 수행 중입니다...")
            
            # 예측을 위한 특성 선택
            X_pred = features_df[feature_cols]
            
            # 예측 수행 (확률 예측)
            pred_proba = model.predict(X_pred, num_iteration=model.best_iteration)
            
            # 확률을 -1(매도), 0(보유), 1(매수)로 변환
            # 0.4 미만: 매도(-1), 0.6 초과: 매수(1), 그 외: 보유(0)
            predictions = np.zeros_like(pred_proba, dtype=int)
            predictions[pred_proba < 0.4] = -1  # 매도
            predictions[pred_proba > 0.6] = 1    # 매수
            
            # 예측 결과를 데이터프레임에 추가
            features_df['prediction'] = predictions
            features_df['pred_proba'] = pred_proba  # 확률도 저장
            
            # 예측 분포 출력
            pred_counts = pd.Series(predictions).value_counts().sort_index()
            logger.info("예측 분포:")
            for val, count in pred_counts.items():
                action = {1: '매수', 0: '보유', -1: '매도'}.get(val, '알 수 없음')
                logger.info(f"  {action}({val}): {count}개 ({(count/len(predictions)*100):.1f}%)")
            
            logger.info(f"예측이 완료되었습니다. 총 {len(predictions)}개 캔들에 대한 예측 생성")
            
        except Exception as e:
            logger.error(f"예측 중 오류가 발생했습니다: {e}", exc_info=True)
            raise
        
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
        # 백테스트 기간 로깅
        logger.info(f"백테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"초기 자본: {initial_capital:,.2f} USDT, 레버리지: {leverage}x")
        
        # 백테스트 엔진 초기화
        backtest = BacktestEngine(
            initial_capital=initial_capital,
            commission=0.0004,  # 바이낸스 수수료 0.04%
            leverage=leverage,
            slippage=0.0005  # 0.05% 슬리피지
        )
        
        # 백테스트에 데이터와 예측 결과 전달
        # 필요한 컬럼만 선택하여 전달 (Backtrader가 예상하는 형식으로 변환)
        
        # prediction 컬럼이 없으면 랜덤 예측값 생성 (-1, 0, 1)
        if 'prediction' not in features_df.columns:
            logger.warning("'prediction' 컬럼이 없어 랜덤 예측값을 생성합니다.")
            features_df['prediction'] = np.random.choice([-1, 0, 1], size=len(features_df))
        
        backtest_data = features_df[['open', 'high', 'low', 'close', 'volume', 'prediction']].copy()
        backtest_data.index = pd.to_datetime(features_df.index)
        backtest_data = backtest_data.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'prediction': 'prediction'
        })
        
        # NaN 값이 있는 경우 0으로 대체 (보유 신호)
        backtest_data['prediction'] = backtest_data['prediction'].fillna(0).astype(int)
        
        # 백테스트에 데이터 로드
        backtest.load_data(
            data=backtest_data,
            price_col='close',
            prediction_col='prediction',
            datetime_col=None  # 인덱스가 이미 datetime으로 설정되어 있으므로 None으로 지정
        )
        
        # 백테스트 실행
        logger.info("백테스트를 시작합니다...")
        results = backtest.run_backtest()
        
        if not results or 'kpis' not in results:
            raise ValueError("백테스트 결과를 가져오는 데 실패했습니다.")
        
        # 결과 출력
        kpis = results['kpis']
        logger.info("\n" + "="*70)
        logger.info(f"{'BACKTEST RESULTS':^70}")
        logger.info("="*70)
        logger.info(f"{'Initial Portfolio Value:':<30} {initial_capital:>20,.2f} USDT")
        logger.info(f"{'Final Portfolio Value:':<30} {results.get('final_value', initial_capital):>20,.2f} USDT")
        logger.info(f"{'Total Return:':<30} {results.get('return_pct', 0):>20.2f}%")
        logger.info("-"*70)
        logger.info(f"{'Sharpe Ratio:':<30} {kpis.get('sharpe_ratio', 'N/A'):>20.2f}")
        logger.info(f"{'Max Drawdown:':<30} {kpis.get('max_drawdown_pct', 'N/A'):>19.2f}%")
        logger.info(f"{'Profit Factor:':<30} {kpis.get('profit_factor', 'N/A'):>20.2f}")
        logger.info(f"{'Win Rate:':<30} {kpis.get('win_rate', 0):>19.2f}%")
        logger.info(f"{'Total Trades:':<30} {kpis.get('number_of_trades', 0):>20}")
        logger.info("="*70)
        
        # 결과 저장을 위한 데이터프레임 생성
        result_data = {
            'symbol': [symbol],
            'timeframe': [timeframe],
            'start_date': [start_date.strftime("%Y-%m-%d")],
            'end_date': [end_date.strftime("%Y-%m-%d")],
            'initial_capital': [initial_capital],
            'final_value': [results.get('final_value', initial_capital)],
            'total_return_pct': [results.get('return_pct', 0)],
            'sharpe_ratio': [kpis.get('sharpe_ratio', None)],
            'max_drawdown_pct': [kpis.get('max_drawdown_pct', None)],
            'profit_factor': [kpis.get('profit_factor', None)],
            'win_rate_pct': [kpis.get('win_rate', 0)],
            'total_trades': [kpis.get('number_of_trades', 0)],
            'leverage': [leverage],
            'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        
        result_df = pd.DataFrame(result_data)
        
        # 결과를 CSV 파일로 저장
        result_file = RESULTS_DIR / f"backtest_results_{symbol.replace('/', '_')}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        result_df.to_csv(result_file, index=False)
        logger.info(f"백테스트 결과가 저장되었습니다: {result_file}")
        
        # 백테스트 결과 반환 (필요한 경우)
        return {
            'success': True,
            'result_file': str(result_file),
            'metrics': {
                'final_value': results.get('final_value', initial_capital),
                'total_return_pct': results.get('return_pct', 0),
                'sharpe_ratio': kpis.get('sharpe_ratio', None),
                'max_drawdown_pct': kpis.get('max_drawdown_pct', None),
                'win_rate_pct': kpis.get('win_rate', 0),
                'total_trades': kpis.get('number_of_trades', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"백테스트 실행 중 오류가 발생했습니다: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    logger.info("파이프라인 실행이 완료되었습니다.")

def main():
    try:
        # 기본 파라미터로 파이프라인 실행
        run_pipeline(
            symbol="BTC/USDT",
            timeframe="4h",
            days=365,  # 1년치 데이터 사용
            initial_capital=10000.0,  # 초기 자본 10,000 USDT
            leverage=3.0  # 3배 레버리지
        )
    except Exception as e:
        logger.error(f"파이프라인 실행 중 치명적 오류 발생: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
