"""
R3000 量化 MVP 系统 - 全局配置文件
上帝视角标注 + 特征提取 + 模式挖掘 + 遗传算法优化
"""

# ==================== 数据配置 ====================
DATA_CONFIG = {
    "DATA_FILE": "btcusdt_1m.parquet",
    "SAMPLE_SIZE": 50000,       # 每次采样的 K 线数量 (50000-100000)
    "RANDOM_SEED": None,        # 随机种子，None 表示每次随机
    "WARMUP_BARS": 200,         # 预热期（用于计算指标）
}

# ==================== 标注配置 ====================
LABELING_CONFIG = {
    # ZigZag 自适应参数
    "ATR_PERIOD": 14,           # ATR 周期
    "SWING_FACTOR": 2.0,        # 摆动阈值 = SWING_FACTOR * ATR
    
    # 交易约束
    "MIN_HOLD_PERIODS": 30,     # 最小持仓周期（1分钟K线数）
    "MIN_RISK_REWARD": 1.5,     # 最小风险收益比
    "MAX_HOLD_PERIODS": 480,    # 最大持仓周期（8小时）
    
    # 标注类型
    "LABEL_BUY": 1,
    "LABEL_SELL": -1,
    "LABEL_HOLD": 0,
    
    # 高低点窗口（用于本地极值标注）
    "SWING_WINDOW": 5,
}

# ==================== 特征配置 ====================
FEATURE_CONFIG = {
    "NUM_FEATURES": 52,         # 特征维度
    
    # MA 周期
    "MA_PERIODS": [5, 10, 20, 60],
    "EMA_PERIODS": [12, 26],
    
    # 动量指标周期
    "RSI_PERIOD": 14,
    "RSI_FAST_PERIOD": 6,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "KDJ_PERIOD": 9,
    "ROC_PERIOD": 10,
    
    # 波动率指标
    "ATR_PERIOD": 14,
    "BOLL_PERIOD": 20,
    "BOLL_STD": 2,
    
    # 成交量指标
    "VOLUME_MA_PERIODS": [5, 20],
    
    # ADX 指标
    "ADX_PERIOD": 14,
    
    # 特征归一化范围
    "FEATURE_CLIP_MIN": -1.0,
    "FEATURE_CLIP_MAX": 1.0,
}

# ==================== 多时间框架配置 ====================
MTF_CONFIG = {
    "TIMEFRAMES": {
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
    },
    # 需要计算的多时间框架特征
    "MTF_FEATURES": ["ema_slope", "rsi", "trend_align", "volatility"],
}

# ==================== 模式挖掘配置 ====================
PATTERN_CONFIG = {
    # 特征重要性分析
    "RF_N_ESTIMATORS": 100,     # 随机森林树数量
    "RF_MAX_DEPTH": 10,         # 最大深度
    
    # 生存分析
    "SURVIVAL_BINS": 20,        # 直方图分箱数
    "PROFIT_QUANTILES": [0.1, 0.25, 0.5, 0.75, 0.9],  # 分位数
}

# ==================== 遗传算法配置 ====================
GA_CONFIG = {
    "POPULATION_SIZE": 50,      # 种群大小
    "ELITE_RATIO": 0.1,         # 精英保留率
    "MUTATION_RATE": 0.15,      # 变异率
    "MUTATION_STRENGTH": 0.1,   # 变异强度
    "CROSSOVER_RATE": 0.7,      # 交叉率
    "MAX_GENERATIONS": 100,     # 最大进化代数
    "EARLY_STOP_GENERATIONS": 20,  # 早停代数（无改进则停止）
    
    # 适应度函数权重
    "FITNESS_WEIGHTS": {
        "sharpe_ratio": 0.4,
        "max_drawdown": 0.3,
        "win_rate": 0.2,
        "profit_factor": 0.1,
    },
}

# ==================== 回测配置 ====================
BACKTEST_CONFIG = {
    "INITIAL_CAPITAL": 10000,   # 初始资金 (USDT)
    "LEVERAGE": 10,             # 杠杆倍数
    "FEE_RATE": 0.0004,         # 手续费率 0.04%
    "SLIPPAGE": 0.0002,         # 滑点 0.02%
    "POSITION_SIZE_PCT": 0.1,   # 单次仓位比例 10%
    "STOP_LOSS_PCT": 0.02,      # 默认止损 2%
    "TAKE_PROFIT_PCT": 0.03,    # 默认止盈 3%
    "TP_SP_LOOKBACK": 20,       # TP/SP 近端高低点窗口
}

# ==================== 标注回测配置（基于标记点） ====================
LABEL_BACKTEST_CONFIG = {
    "INITIAL_CAPITAL": 5000,    # 本金 (USDT)
    "LEVERAGE": 10,             # 固定 10x
    "POSITION_SIZE_PCT": 1.0,   # 全仓保证金
    "FEE_RATE": 0.0004,         # 手续费
    "SLIPPAGE": 0.0002,         # 滑点
}

# ==================== 向量空间配置 ====================
VECTOR_SPACE_CONFIG = {
    "K_NEIGHBORS": 5,            # 偏离指数计算时取最近的 k 个点
    "MIN_CLOUD_POINTS": 3,       # 点云最少多少个点才计算偏离指数
    "ENTRY_SIM_THRESHOLD": 70.0, # 入场相似度阈值 (%)
    "EXIT_SIM_THRESHOLD": 70.0,  # 出场相似度阈值 (%)
    "LOST_THRESHOLD": 20.0,      # 迷失信号阈值 (%) - entry和exit都低于此值
    "EXIT_CONFIRM_BARS": 3,      # 出场信号需连续确认的K线数
    "PRECISION": 3,              # 坐标精度（小数位）
    "ENTRY_CONFIRM_PCT": 0.0001, # 入场价格确认比例 (0.01%)
    "TRIGGER_TIMEOUT_BARS": 5,   # 信号触发超时（K线数）
    # GA权重搜索范围
    "WEIGHT_MIN": -10.0,
    "WEIGHT_MAX": 10.0,
}

# ==================== 轨迹匹配配置 ====================
TRAJECTORY_CONFIG = {
    "PRE_ENTRY_WINDOW": 60,      # 入场前回看窗口（K线数）
    "PRE_EXIT_WINDOW": 30,       # 离场前回看窗口（K线数）
    "COSINE_TOP_K": 5,           # 余弦粗筛保留候选数量（降低以加速DTW）
    "DTW_RADIUS": 5,             # FastDTW半径约束（加速计算）
    "MIN_PROFIT_PCT": 0.5,       # 只提取收益率>0.5%的交易作为模板
    "FEATURE_DIM": 32,           # 特征维度 (16+10+6)
    # 匹配阈值默认值（由贝叶斯优化）
    "COSINE_THRESHOLD": 0.95,    # 余弦相似度阈值 (95%)
    "DTW_THRESHOLD": 0.5,        # DTW归一化距离阈值（越小越严格）
    "HOLD_DIVERGENCE_LIMIT": 0.7,# 持仓偏离上限
    "EXIT_MATCH_THRESHOLD": 0.5, # 离场匹配阈值
    # 优化加速参数
    "EVAL_SKIP_BARS": 5,         # 模拟交易时每N根K线评估一次（加速）
}

# ==================== Walk-Forward验证配置 ====================
WALK_FORWARD_CONFIG = {
    "TRAIN_RATIO": 0.6,          # 训练集比例
    "VAL_RATIO": 0.2,            # 验证集比例
    "TEST_RATIO": 0.2,           # 测试集比例
    "N_FOLDS": 3,                # 滑动窗口折数
    "STEP_SIZE": 5000,           # 每折滑动的K线数
    "MIN_TRAIN_TRADES": 30,      # 训练集最少交易数
    "MIN_VAL_TRADES": 10,        # 验证集最少交易数

    # 模板评估筛选参数
    "EVAL_MIN_MATCHES": 2,       # 最小匹配次数（低于此为"待观察"）
    "EVAL_MIN_WIN_RATE": 0.45,   # 最小胜率（配合盈亏比>1，45%即正期望）
    "EVAL_MIN_AVG_PROFIT": 0.0,  # 最小平均收益（默认0）
    "EVAL_USE_EXPECTED_PROFIT": True,  # 是否使用期望收益(avg_profit>0)作为主要合格标准
    "BATCH_ROUND_WORKERS": 2,    # 批量WF并行轮次线程数（1=串行）
    "BATCH_DEFAULT_ROUNDS": 20,  # 批量WF默认轮数（更多轮次=更充分验证）
}

# ==================== 记忆持久化配置 ====================
MEMORY_CONFIG = {
    "MEMORY_DIR": "data/memory",      # 记忆存储目录
    "AUTO_SAVE": True,                # 回测完成后自动保存记忆
    "AUTO_LOAD": True,                # 启动时自动加载最新记忆
    "MERGE_ON_LOAD": True,            # 加载时合并（而非覆盖）
    "DEDUPLICATE": True,              # 合并时去重
    "MAX_MEMORY_FILES": 50,           # 最多保留的记忆文件数
}

# ==================== 原型聚类配置 ====================
PROTOTYPE_CONFIG = {
    "N_CLUSTERS_LONG": 30,            # LONG 方向聚类数
    "N_CLUSTERS_SHORT": 30,           # SHORT 方向聚类数
    "MIN_CLUSTER_SIZE": 3,            # 最小簇大小（小于此丢弃）
    "PROTOTYPE_DIR": "data/prototypes",  # 原型存储目录
    "AUTO_LOAD_PROTOTYPE": True,      # 启动时自动加载最新原型库
    # 原型匹配参数
    "COSINE_THRESHOLD": 0.95,         # 余弦相似度阈值 (95%，更严格)
    "MIN_PROTOTYPES_AGREE": 2,        # 最少原型同意数（投票）
    "HOLDING_SAFE_THRESHOLD": 0.7,    # 持仓健康阈值
    "HOLDING_ALERT_THRESHOLD": 0.5,   # 持仓警告阈值
    "HOLDING_DANGER_THRESHOLD": 0.3,  # 持仓危险阈值
}

# ==================== 市场状态分类配置 ====================
MARKET_REGIME_CONFIG = {
    "DIR_STRONG_THRESHOLD": 0.008,       # 方向强趋势阈值 (0.8%)
    "DIR_WEAK_THRESHOLD": 0.001,         # 方向弱趋势阈值 (0.1%，更灵敏)
    "STRENGTH_STRONG_THRESHOLD": 0.006,  # 强度阈值 (振幅/均价 > 0.6%)
    "LOOKBACK_SWINGS": 6,               # 回看摆动点数量 (增加到6个，更全面)
}

# ==================== UI 配置 ====================
UI_CONFIG = {
    "WINDOW_TITLE": "R3000 量化 MVP 系统",
    "WINDOW_WIDTH": 1600,
    "WINDOW_HEIGHT": 900,
    
    # 深色主题
    "THEME_BACKGROUND": "#1e1e1e",
    "THEME_SURFACE": "#252526",
    "THEME_TEXT": "#cccccc",
    "THEME_ACCENT": "#007acc",

    # 字体大小（适配高分辨率屏幕，可调大）
    "FONT_SIZE_NORMAL": 12,       # 正常文字
    "FONT_SIZE_SMALL": 11,        # 小文字（表格等）
    "FONT_SIZE_LARGE": 14,        # 大标题
    "FONT_SIZE_XLARGE": 16,       # 特大（当前状态等）
    
    # 图表配置
    "CHART_CANDLE_WIDTH": 0.8,
    "CHART_UP_COLOR": "#089981",      # 币安绿
    "CHART_DOWN_COLOR": "#f23645",    # 币安红
    "CHART_BACKGROUND": "#131722",    # 图表背景（TradingView风格）
    "CHART_GRID_COLOR": "#363a45",    # 网格颜色
    
    # 信号标记颜色
    "CHART_LONG_ENTRY_COLOR": "#00E676",   # 做多入场（亮绿）
    "CHART_LONG_EXIT_COLOR": "#69F0AE",    # 做多出场（浅绿）
    "CHART_SHORT_ENTRY_COLOR": "#FF5252",  # 做空入场（亮红）
    "CHART_SHORT_EXIT_COLOR": "#FF8A80",   # 做空出场（浅红）
    "CHART_TAKE_PROFIT_COLOR": "#FFD54F",  # 止盈（黄色）
    "CHART_STOP_LOSS_COLOR": "#FF1744",    # 止损（红色）
    
    # 播放速度配置
    "DEFAULT_SPEED": 10,              # 默认速度 10x
    "MIN_SPEED": 1,                   # 最小速度 1x
    "MAX_SPEED": 100,                 # 最大速度 100x
    
    # 刷新配置
    "CHART_FPS": 30,
}

# ==================== 模拟交易配置 ====================
PAPER_TRADING_CONFIG = {
    "DEFAULT_SYMBOL": "BTCUSDT",
    "DEFAULT_INTERVAL": "1m",
    "DEFAULT_BALANCE": 5000.0,
    "DEFAULT_LEVERAGE": 10,
    # 数据环境：模拟盘默认用 Futures Testnet
    "USE_TESTNET": True,
    "MARKET_TYPE": "futures",  # "spot" / "futures"
    
    # 匹配参数
    "COSINE_THRESHOLD": 0.95,         # 余弦相似度阈值 (95%)
    "DTW_THRESHOLD": 0.5,
    "MIN_TEMPLATES_AGREE": 1,
    
    # 止盈止损
    "STOP_LOSS_ATR": 2.0,
    "TAKE_PROFIT_ATR": 3.0,
    "MAX_HOLD_BARS": 240,
    
    # 动态追踪
    "HOLD_SAFE_THRESHOLD": 0.7,
    "HOLD_ALERT_THRESHOLD": 0.5,
    "HOLD_DERAIL_THRESHOLD": 0.3,
    "HOLD_CHECK_INTERVAL": 3,
    
    # 数据目录
    "HISTORY_DIR": "data/paper_trading",
    "VERIFIED_DIR": "data/sim_verified",
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    "SAVE_LABELS": True,
    "SAVE_FEATURES": True,
    "SAVE_PATTERNS": True,
    "SAVE_GA_RESULTS": True,
    "OUTPUT_DIR": "data/output",
    "VERBOSE": True,
}
