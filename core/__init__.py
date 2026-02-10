"""
R3000 核心模块
"""
from .data_loader import DataLoader
from .labeler import GodViewLabeler
from .features import FeatureExtractor
from .pattern_miner import PatternMiner
from .genetic_optimizer import GeneticOptimizer
from .backtester import Backtester
from .market_regime import MarketRegimeClassifier, MarketRegime
from .feature_vector import FeatureVectorEngine, AxisWeights
from .vector_memory import VectorMemory
from .ga_weight_optimizer import GAWeightOptimizer
# 轨迹匹配模块
from .trajectory_engine import TrajectoryTemplate, TrajectoryMemory, extract_current_trajectory
from .trajectory_matcher import TrajectoryMatcher, MatchResult, batch_cosine_similarity
from .ga_trading_optimizer import GATradingOptimizer, TradingParams, SimulatedTradeResult
from .walk_forward import WalkForwardValidator, WalkForwardResult, FoldResult

__all__ = [
    'DataLoader',
    'GodViewLabeler', 
    'FeatureExtractor',
    'PatternMiner',
    'GeneticOptimizer',
    'Backtester',
    'MarketRegimeClassifier',
    'MarketRegime',
    'FeatureVectorEngine',
    'AxisWeights',
    'VectorMemory',
    'GAWeightOptimizer',
    # 轨迹匹配
    'TrajectoryTemplate',
    'TrajectoryMemory',
    'extract_current_trajectory',
    'TrajectoryMatcher',
    'MatchResult',
    'batch_cosine_similarity',
    'GATradingOptimizer',
    'TradingParams',
    'SimulatedTradeResult',
    'WalkForwardValidator',
    'WalkForwardResult',
    'FoldResult',
]
