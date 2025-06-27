"""
바이낸스 선물 거래소에서 4시간봉 OHLCV 데이터를 수집하는 모듈입니다.

핵심 기능:
- 지정된 기간의 4H OHLCV 데이터 수집 (BTC/USDT)
- 수집된 데이터를 Parquet 형식으로 저장
- 중복 데이터 처리 및 데이터 무결성 검증
"""
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import ccxt
from loguru import logger
from tqdm import tqdm

# 로깅 설정
logger.add(
    "logs/data_collection.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

class BinanceDataCollector:
    """바이낸스 선물 거래소에서 4시간봉 데이터를 수집하는 클래스"""
    
    def __init__(self, base_dir: str = "data/raw"):
        """
        초기화 함수
        
        Args:
            base_dir (str): 데이터 저장 기본 경로
        """
        self.exchange = ccxt.binanceusdm({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # 선물 거래
            },
        })
        
        # 디렉토리 생성
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 거래 쌍 설정 (BTC/USDT)
        self.symbol = 'BTC/USDT'
        self.timeframe = '4h'  # 4시간봉
        
    def fetch_ohlcv(
        self, 
        start_date: str, 
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        지정된 기간의 OHLCV 데이터를 가져옵니다.
        
        Args:
            start_date (str): 시작 날짜 (YYYY-MM-DD 형식)
            end_date (str, optional): 종료 날짜 (None인 경우 현재 시점까지)
            limit (int): 한 번의 요청으로 가져올 최대 데이터 포인트 수 (기본값: 1000)
            
        Returns:
            pd.DataFrame: OHLCV 데이터프레임 (timestamp, open, high, low, close, volume)
        """
        since = self._parse_date_to_timestamp(start_date)
        if end_date:
            end_timestamp = self._parse_date_to_timestamp(end_date)
        else:
            end_timestamp = int(time.time() * 1000)
            
        all_ohlcv = []
        current_since = since
        
        with tqdm(desc="데이터 수집 중", unit="캔들") as pbar:
            while current_since < end_timestamp:
                try:
                    # API 호출
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        since=current_since,
                        limit=limit
                    )
                    
                    if not ohlcv:
                        break
                        
                    # 데이터프레임으로 변환
                    df = pd.DataFrame(
                        ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    
                    # 타임스탬프를 datetime으로 변환
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    all_ohlcv.append(df)
                    
                    # 다음 페이지를 위해 since 업데이트 (마지막 타임스탬프 + 1ms)
                    current_since = df['timestamp'].iloc[-1] + 1
                    
                    # 진행 상황 업데이트
                    pbar.update(len(df))
                    
                    # Rate limit 방지
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                except Exception as e:
                    logger.error(f"Error fetching OHLCV data: {e}")
                    time.sleep(5)  # 에러 발생 시 잠시 대기 후 재시도
        
        if not all_ohlcv:
            return pd.DataFrame()
            
        # 모든 데이터프레임 병합
        result_df = pd.concat(all_ohlcv, ignore_index=True)
        
        # 중복 제거 및 정렬
        result_df = result_df.drop_duplicates(subset=['timestamp'])
        result_df = result_df.sort_values('timestamp').reset_index(drop=True)
        
        return result_df
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """
        데이터프레임을 Parquet 형식으로 저장합니다.
        
        Args:
            df (pd.DataFrame): 저장할 데이터프레임
            filename (str): 저장할 파일명 (확장자 제외)
        """
        if df.empty:
            logger.warning("데이터프레임이 비어있어 저장하지 않습니다.")
            return
            
        # 저장 경로 생성
        filepath = os.path.join(self.base_dir, f"{filename}.parquet")
        
        try:
            # Parquet로 저장
            df.to_parquet(filepath, index=False)
            logger.info(f"데이터가 성공적으로 저장되었습니다: {filepath}")
        except Exception as e:
            logger.error(f"데이터 저장 중 오류 발생: {e}")
    
    def _parse_date_to_timestamp(self, date_str: str) -> int:
        """
        날짜 문자열을 밀리초 단위의 타임스탬프로 변환합니다.
        
        Args:
            date_str (str): YYYY-MM-DD 형식의 날짜 문자열
            
        Returns:
            int: 밀리초 단위의 타임스탬프
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)

def main():
    """메인 실행 함수"""
    # 데이터 수집기 초기화
    collector = BinanceDataCollector()
    
    # 최근 1년치 데이터 수집 (예시)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    logger.info(f"{start_date}부터 {end_date}까지의 데이터를 수집합니다...")
    
    # 데이터 수집
    df = collector.fetch_ohlcv(start_date=start_date, end_date=end_date)
    
    if not df.empty:
        # 저장할 파일명 생성 (예: btc_usdt_4h_20230101_20231231)
        start_str = df['datetime'].min().strftime("%Y%m%d")
        end_str = df['datetime'].max().strftime("%Y%m%d")
        filename = f"btc_usdt_4h_{start_str}_{end_str}"
        
        # 데이터 저장
        collector.save_to_parquet(df, filename)
        
        # 요약 정보 로깅
        logger.info(f"수집된 데이터 요약:")
        logger.info(f"- 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
        logger.info(f"- 총 캔들 수: {len(df)}")
        logger.info(f"- 시작 가격: {df['open'].iloc[0]:.2f} USDT")
        logger.info(f"- 종료 가격: {df['close'].iloc[-1]:.2f} USDT")
        logger.info(f"- 변동률: {((df['close'].iloc[-1] / df['open'].iloc[0] - 1) * 100):.2f}%")
    else:
        logger.warning("수집된 데이터가 없습니다.")

if __name__ == "__main__":
    main()
