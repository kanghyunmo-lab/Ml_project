"""
백테스팅 관련 모듈을 포함하는 패키지입니다.

이 패키지는 트레이딩 전략의 성능을 평가하기 위한 백테스팅 기능을 제공합니다.
"""

from .engine import BacktestEngine, TripleBarrierStrategy

__all__ = ['BacktestEngine', 'TripleBarrierStrategy']
