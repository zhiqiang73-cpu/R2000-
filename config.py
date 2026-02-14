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

# ==================== 实盘风控配置 ====================
LIVE_RISK_CONFIG = {
    "MAX_DRAWDOWN_PCT": None,   # 已禁用，不再自动锁定开仓
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
    "ENTRY_CONFIRM_PCT": 0.0008, # 入场价格确认比例 (0.08%)，要求价格先朝目标方向走~$53(BTC)才成交
    "TRIGGER_TIMEOUT_BARS": 5,   # 信号触发超时（K线数）
    "ENTRY_COOLDOWN_SEC": 8,     # 连续开仓冷却时间（秒）
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
    "STRENGTH_STRONG_THRESHOLD": 0.006,  # 强度阈值 (振幅/均价 > 0.6%)
    "LOOKBACK_SWINGS": 4,               # 回看摆动点数量（只看最近走势）
    # 短期趋势修正（让 regime 更贴近 K 线走势）
    "SHORT_TREND_LOOKBACK": 6,          # 近 N 根 K 线方向（更聚焦当前）
    "SHORT_TREND_THRESHOLD": 0.002,     # 0.2% 视为短期趋势明显
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
    
    # ── 止损保护参数（防止正常波动被扫损）──
    "MIN_SL_PCT": 0.005,        # 最小止损距离 0.5%（BTC≈$475），挡住1分钟正常波动
    "ATR_SL_MULTIPLIER": 3.0,   # ATR 止损倍数（越大越宽松），原来 2.0 太紧
    "SL_PROTECTION_SEC": 60,    # 新仓保护期（秒），保护期内止损暂缓，原来 8 秒太短
    
    # ── 前3根K线紧急止损守卫 (Early Exit Guard) ──
    # 问题：前3根保护期内只有硬止损保护，形成致命盲区（3根就能亏3-4%）
    # 方案1：亏损超阈值 → 紧急平仓
    # 方案2：原始市场状态与持仓方向冲突 → 收紧止损至入场价附近
    "EARLY_EXIT_ADVERSE_PCT": 1.5,     # 保护期内亏损超此%（杠杆后）→ 紧急平仓
    "EARLY_EXIT_TIGHTEN_PCT": 0.005,   # 市场冲突时止损收紧到入场价±此%（0.5%）
    
    # ── 反手单 (Stop-and-Reverse) ──
    # 止损说明方向判断错误，自动反手做反方向
    "REVERSE_ON_STOPLOSS": True,      # 是否启用止损反手
    "REVERSE_MAX_COUNT": 1,           # 最多连续反手次数（防止来回被扫）
    "REVERSE_BLOCK_SAME_DIR_SEC": 300,  # 止损后禁止同方向入场的时间（秒）
    
    # 动态追踪
    "HOLD_SAFE_THRESHOLD": 0.7,
    "HOLD_ALERT_THRESHOLD": 0.5,
    "HOLD_DERAIL_THRESHOLD": 0.3,
    
    # ── 追踪止损阶段阈值（降低以更早锁定利润）──
    "TRAILING_STAGE1_PCT": 1.0,       # 阶段1(保本)激活阈值：峰值利润>=1%（原1.5%）
    "TRAILING_STAGE2_PCT": 2.0,       # 阶段2(锁利40%)激活阈值：峰值利润>=2%（原3%）
    "TRAILING_STAGE3_PCT": 3.5,       # 阶段3(紧追60%)激活阈值：峰值利润>=3.5%（原5%）
    "TRAILING_LOCK_PCT_STAGE2": 0.50, # 阶段2锁定峰值利润的比例（50%，原40%）
    "TRAILING_LOCK_PCT_STAGE3": 0.70, # 阶段3锁定峰值利润的比例（70%，原60%）
    
    # ── 价格动量衰减离场（新增）──
    # 当价格接近峰值但动能衰减时主动离场
    "MOMENTUM_EXIT_ENABLED": True,     # 是否启用动量衰减离场
    "MOMENTUM_MIN_PROFIT_PCT": 1.5,    # 最低利润阈值（至少盈利1.5%才检测）
    "MOMENTUM_LOOKBACK_BARS": 3,       # 动量检测回看K线数
    "MOMENTUM_DECAY_THRESHOLD": 0.5,   # K线实体缩小阈值（当前<峰值的50%视为衰减）
    "MOMENTUM_PEAK_RETRACEMENT": 0.8,  # 从峰值回撤阈值（回撤80%的利润触发）
    "HOLD_CHECK_INTERVAL": 3,
    
    # ── 离场信号学习系统（自适应优化）──
    "EXIT_LEARNING_ENABLED": True,     # 是否启用离场信号学习
    "EXIT_LEARNING_STATE_FILE": "data/exit_learning_state.json",  # 持久化文件路径
    "EXIT_LEARNING_DECAY_ENABLED": True,  # 是否启用时间衰减
    "EXIT_LEARNING_DECAY_HOURS": 48.0,    # 衰减间隔（小时）
    "EXIT_LEARNING_DECAY_FACTOR": 0.95,   # 衰减因子（0-1）
    "EXIT_LEARNING_MIN_SAMPLES": 10,      # 最少样本数（少于此值使用默认策略）

    # 实时决策频率（秒）
    "REALTIME_DECISION_SEC": 0.05,
    # 是否允许在未收线K线中进行入场决策
    "REALTIME_ENTRY_ENABLED": True,
    # REST轮询频率（秒，过高可能被限流）
    "REALTIME_REST_POLL_SEC": 0.05,
    # UI刷新频率（毫秒）
    "REALTIME_UI_REFRESH_MS": 50,
    
    # 数据目录
    "HISTORY_DIR": "data/paper_trading",
    "VERIFIED_DIR": "data/sim_verified",

    # 离场限价 IOC 价格缓冲（%），更激进 = 更高 IOC 成交率、减少市价降级（省 Taker 费）
    # 0.001=0.1%（原值）, 0.003=0.3%（推荐）, 0.005=0.5%（极端波动）
    "EXIT_IOC_BUFFER_PCT": 0.003,
    
    # ── 贝叶斯交易过滤器（Thompson Sampling + 凯利公式）──
    "BAYESIAN_ENABLED": True,              # 是否启用贝叶斯门控
    "BAYESIAN_PRIOR_STRENGTH": 10.0,       # 先验强度（历史回测数据相当于多少笔实盘交易）
    "BAYESIAN_MIN_WIN_RATE": 0.40,         # 最低胜率阈值（低于此值拒绝交易）
    "BAYESIAN_THOMPSON_SAMPLING": True,    # 是否使用 Thompson Sampling（探索与利用平衡）
    "BAYESIAN_DECAY_ENABLED": True,        # 是否启用时间衰减（适应市场变化）
    "BAYESIAN_DECAY_HOURS": 24.0,          # 衰减间隔（小时）
    "BAYESIAN_DECAY_FACTOR": 0.95,         # 衰减因子（0-1，越小遗忘越快）
    "BAYESIAN_STATE_FILE": "data/bayesian_state.json",  # 持久化文件路径
    
    # ── 凯利公式动态仓位管理 ──
    "KELLY_ENABLED": True,                 # 是否启用凯利公式动态仓位
    "KELLY_FRACTION": 0.25,                # 凯利分数（0.25=四分之一凯利，推荐范围 0.25-0.5）
    "KELLY_MAX_POSITION": 0.5,             # 凯利仓位上限（50% 本金）
    "KELLY_MIN_POSITION": 0.05,            # 凯利仓位下限（5% 本金，样本不足时保守试探）
    "KELLY_MIN_SAMPLES": 5,                # 凯利计算最少样本数（少于此值用最小仓位）
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
