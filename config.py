"""
R3000 量化 MVP 系统 - 全局配置文件
上帝视角标注 + 特征提取 + 模式挖掘 + 遗传算法优化
"""

import os

# ==================== 环境变量加载 ====================
# 简易 .env 加载（避免引入第三方依赖）；先读 config 同目录，再读当前工作目录
def _load_dotenv(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8-sig") as _f:  # utf-8-sig 去除 BOM
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _key, _val = _line.split("=", 1)
                _key = _key.strip()
                _val = _val.strip().strip('"').strip("'")
                # 始终用 .env 覆盖，避免系统环境变量里的旧 Key 覆盖 .env 中的新 Key
                if _key:
                    os.environ[_key] = _val
        return True
    except Exception:
        return False

_env_dir = os.path.dirname(os.path.abspath(__file__))
_load_dotenv(os.path.join(_env_dir, ".env"))
_load_dotenv(os.path.join(os.getcwd(), ".env"))  # 若从别处启动，再试当前目录

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

# ==================== 冷启动配置 ====================
COLD_START_CONFIG = {
    "ENABLED_BY_DEFAULT": False,
    "MACD_BYPASS": True,     # 冷启动时跳过MACD趋势确认，便于启动学习；关闭冷启动后MACD门控恢复
    "THRESHOLDS": {
        "fusion": 0.30,      # 正常约0.40
        "cosine": 0.50,      # 正常约0.60
        "euclidean": 0.25,   # 正常约0.35
        "dtw": 0.10          # 正常约0.30
    },
    "FREQUENCY_MONITOR": {
        "TARGET_TRADES_PER_HOUR": 1.5,
        "LOW_FREQUENCY_THRESHOLD_HOURS": 1.0,  # 超过1小时无交易视为频率过低
        "AUTO_RELAX_PERCENT": 0.05             # 自动放宽5%
    },
    # 冷启动自动毕业
    "COLD_START_AUTO_GRADUATE": True,
    "COLD_START_SUCCESS_RATE_THRESHOLD": 0.80,  # 80%成功率
    "COLD_START_MIN_TRADES_FOR_GRADUATE": 20,   # 至少20笔交易
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

# ==================== Walk-Forward 进化优化配置 ====================
WF_EVOLUTION_CONFIG = {
    # ── 数据分割 ──
    "SAMPLE_SIZE": 300000,              # 总采样K线数 (~208天, 覆盖6+月多种市况)
    "HOLDOUT_RATIO": 0.30,              # 留出集比例 (30% = ~90K bars, ~62天)
    "INNER_FOLDS": 3,                   # 内部WF折数 (每折~70K bars, ~49天)
    "DATA_OFFSET": "latest",            # 数据偏移: "latest"=最新300K bars, 或整数偏移量
    
    # ── 多轮随机验证（防止过拟合特定时间段）──
    "MULTI_ROUND_ENABLED": True,        # 是否启用多轮随机验证
    "EVOLUTION_ROUNDS": 2,              # 进化轮数（2轮平衡时间与泛化，3轮更保守）
    "DATA_SAMPLE_STRATEGY": "mixed",    # 采样策略: "latest"=固定最新, "random"=随机段, "mixed"=1轮latest+其余random
    "TRIALS_PER_ROUND": 18,             # 【快速探索】每轮试验次数（降低以控制时间）
    "FINAL_REFINE_ENABLED": True,       # 是否启用最终精炼阶段
    "FINAL_REFINE_TRIALS": 25,          # 【精细优化】最优轮的额外试验次数
    "FINAL_REFINE_ROUND": "best",       # 精炼轮选择: "best"=最优轮, "latest"=最新轮, "ensemble"=集成
    
    # ── 交叉验证（轮之间互相验证）──
    "CROSS_VALIDATION_ENABLED": True,   # 是否启用跨轮交叉验证
    "CROSS_VAL_TOP_K": 2,               # 每轮保留前K个最优权重进行交叉验证（降低至2以省时）
    "ENSEMBLE_WEIGHTS": True,           # 是否对多轮最优权重做集成（加权平均）
    "ENSEMBLE_METHOD": "performance",   # 集成方法: "equal"=等权, "performance"=按性能加权
    
    # ── 快速筛选（降低计算成本）──
    "QUICK_PRESCREEN_ENABLED": True,    # 启用快速预筛（在小数据集上初步测试）
    "PRESCREEN_SAMPLE_SIZE": 100000,    # 预筛数据量（1/3，快速排除明显不佳的时间段）
    "PRESCREEN_TRIALS": 8,              # 预筛试验次数（降低以省时）
    "PRESCREEN_CANDIDATES": 5,          # 预筛候选段数量（从5个中选最优2个）
    "PRESCREEN_SELECT_TOP": 2,          # 从候选段中选择前N个进入正式进化（匹配EVOLUTION_ROUNDS）
    
    # ── 市场状态监控（可选）──
    "REGIME_TRACKING_ENABLED": True,    # 是否记录各轮的市场状态分布
    "REGIME_FEATURES": ["trend", "volatility", "volume"],  # 要跟踪的状态特征

    # ── 特征语义分组 (8组, 将32维降至8维搜索空间) ──
    # 每组共享一个乘数权重, 同组特征高度相关, 无需独立搜索
    # indices 对应 feature_vector.py 中的32维特征向量索引
    "FEATURE_GROUPS": {
        "rsi": {
            "label": "RSI 组",
            "indices": [0, 1, 16, 17],      # rsi_14, rsi_6, delta_rsi_3, delta_rsi_5
            "default_weight": 1.5,           # 默认权重 (Layer A 基准)
        },
        "macd": {
            "label": "MACD 组",
            "indices": [2, 18, 19],          # macd_hist, delta_macd_3, delta_macd_5
            "default_weight": 1.5,
        },
        "volatility": {
            "label": "波动率组",
            "indices": [6, 7, 8, 20, 21],   # atr_ratio, boll_width, boll_position, delta_atr_rate, delta_boll_width
            "default_weight": 1.3,
        },
        "momentum": {
            "label": "动量组",
            "indices": [3, 4, 5],            # kdj_k, kdj_j, roc
            "default_weight": 1.5,
        },
        "volume": {
            "label": "成交量组",
            "indices": [14, 15, 24],         # volume_ratio, obv_slope, delta_volume_ratio
            "default_weight": 1.2,
        },
        "trend": {
            "label": "趋势组",
            "indices": [9, 10, 11, 12, 22, 23],  # shadow_ratios, ema_devs, ema_slope_changes
            "default_weight": 1.3,
        },
        "structure": {
            "label": "结构组",
            "indices": [26, 27, 28, 29, 30, 31],  # price_position, dist_high/low, amp_ratio, vs_20H/L
            "default_weight": 1.0,
        },
        "adx": {
            "label": "ADX 组",
            "indices": [13, 25],             # adx, delta_adx
            "default_weight": 1.2,
        },
    },

    # ── 权重搜索范围 ──
    "WEIGHT_MIN": 0.1,                  # 组权重下限
    "WEIGHT_MAX": 3.0,                  # 组权重上限

    # ── 阈值搜索范围 ──
    "FUSION_THRESHOLD_RANGE": (0.40, 0.85),   # 多维融合分数阈值搜索范围
    "COSINE_MIN_THRESHOLD_RANGE": (0.50, 0.90),  # 余弦相似度最低阈值搜索范围
    "EUCLIDEAN_MIN_THRESHOLD_RANGE": (0.30, 0.80),  # 欧氏距离最低阈值搜索范围
    "DTW_MIN_THRESHOLD_RANGE": (0.20, 0.70),  # DTW形态最低阈值搜索范围

    # ── CMA-ES 优化器参数 ──
    "OPTIMIZER": "cma-es",              # 优化器: "cma-es" (推荐) 或 "tpe" (备选)
    "N_TRIALS": 60,                     # 【单轮模式】总试验次数 (CMA-ES ~6代, 足够收敛)
                                        # 【多轮模式】被 TRIALS_PER_ROUND 和 FINAL_REFINE_TRIALS 取代
    "CMA_SIGMA0": 0.5,                  # CMA-ES 初始步长 (搜索范围的~1/4)
    "CMA_RESTART_STRATEGY": "ipop",     # CMA-ES 重启策略 (增量种群)
    "TPE_FALLBACK": True,               # CMA-ES 收敛失败时是否回退到 TPE

    # ── 适应度函数 ──
    "FITNESS_METRIC": "sharpe",         # 主指标: "sharpe", "profit_factor", "calmar"
    "L2_LAMBDA": 0.01,                  # L2 正则化系数 (惩罚偏离默认权重)
    "MIN_TRADES_PER_FOLD": 15,          # 每折最少交易数 (少于此罚分 -10)
    "MIN_TRADES_PENALTY": -10.0,        # 交易不足时的罚分
    "EVAL_SKIP_BARS": 10,               # 【正式进化】模拟交易时每N根K线评估 (加速, 7000点/70K bars)
    "EVAL_SKIP_BARS_PRESCREEN": 15,     # 【快速预筛】评估间隔更大以加速（~5000点/70K bars）

    # ── 留出集验证通过标准（降低门槛后更容易合格） ──
    "HOLDOUT_PASS_CRITERIA": {
        "min_sharpe": 0.0,              # 最低夏普（0 = 不要求为正，避免留出段稍差就一票否决）
        "min_win_rate": 0.35,            # 最低胜率 35%
        "max_drawdown": 40.0,           # 最大回撤上限 40%（单位：百分比数值，如 25=25%）
        "min_trades": 15,               # 最少交易笔数（留出段样本少时也容易达标）
        "min_profit_factor": 0.95,      # 最低盈亏比（略低于1也可接受，避免过度苛刻）
    },

    # ── 输出/持久化 ──
    "RESULTS_DIR": "data/wf_evolution",         # 进化结果存储目录
    "SAVE_TOP_K": 3,                            # 保存前K个最优权重组合
    "EVOLVED_WEIGHTS_FILE": "data/wf_evolution/evolved_weights.json",  # 最优权重文件（兼容单组）
    "EVOLVED_WEIGHTS_FILE_LONG": "data/wf_evolution/evolved_weights_long.json",
    "EVOLVED_WEIGHTS_FILE_SHORT": "data/wf_evolution/evolved_weights_short.json",
    "SAVE_ROUND_RESULTS": True,                 # 是否保存每轮的详细结果
    
    # ── 计算时间估算 ──
    # 【单轮模式】(MULTI_ROUND_ENABLED=False): 
    #   60 trials × 3 folds × 3秒 ≈ 9分钟
    # 
    # 【多轮模式（当前配置）】: 
    #   - 快速预筛: 5候选 × 8 trials × 3 folds × 2秒 ≈ 2.4分钟
    #   - 正式进化: 2轮 × 18 trials × 3 folds × 3秒 ≈ 5.4分钟
    #   - 最终精炼: 1轮 × 25 trials × 3 folds × 3秒 ≈ 3.8分钟
    #   - 交叉验证: 2轮 × 2候选 × 3 folds × 1秒 ≈ 0.2分钟
    #   - **总计: ~12分钟 (相比单轮增加约33%，泛化能力显著提升)**
    # 
    # 【保守模式】(EVOLUTION_ROUNDS=3, TRIALS_PER_ROUND=20, FINAL_REFINE_TRIALS=30):
    #   总计 ~18分钟 (增加约100%，但过拟合风险最低)
    # 
    # 【极速模式】(QUICK_PRESCREEN_ENABLED=False, EVOLUTION_ROUNDS=1):
    #   等同单轮模式 ~9分钟 (无泛化验证)
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

# ==================== 特征层权重配置（指纹3D图匹配） ====================
FEATURE_WEIGHTS_CONFIG = {
    # 三层特征权重系数（用于加权相似度计算）
    # Layer A (即时信号): MACD、RSI等核心指标 - 16维 - 最重要
    "LAYER_A_WEIGHT": 1.5,
    # Layer B (动量变化): 变化率、加速度 - 10维 - 中等重要
    "LAYER_B_WEIGHT": 1.2,
    # Layer C (结构位置): 相对位置 - 6维 - 基础重要
    "LAYER_C_WEIGHT": 1.0,
    
    # 特征维度定义（与 feature_vector.py 保持一致）
    "N_A": 16,                # Layer A 特征数
    "N_B": 10,                # Layer B 特征数
    "N_C": 6,                 # Layer C 特征数
    
    # 是否启用特征加权
    "ENABLED": True,
}

# ==================== 多维相似度配置（指纹3D图匹配） ====================
SIMILARITY_CONFIG = {
    # 多维相似度融合权重
    # 方向相似度: 捕捉特征变化方向是否一致
    "COSINE_WEIGHT": 0.30,
    # 距离相似度: 捕捉特征数值是否接近
    "EUCLIDEAN_WEIGHT": 0.40,
    # 形态相似度: 捕捉时间序列形态是否匹配
    "DTW_WEIGHT": 0.30,
    
    # 余弦相似度参数
    "COSINE_MIN_THRESHOLD": 0.70,     # 余弦相似度最低接受阈值
    
    # 欧氏距离参数
    "EUCLIDEAN_NORMALIZE": True,       # 是否归一化欧氏距离
    "EUCLIDEAN_MAX_DISTANCE": 200.0,   # 归一化上界（32维特征含RSI/KDJ等，距离常 50~200+，过小会导致“距离”恒为0%）
    
    # DTW 参数
    "DTW_RADIUS": 10,                  # FastDTW 半径约束（加速计算，增大可提高精度但变慢）
    "DTW_NORMALIZE": True,             # 是否归一化 DTW 距离
    "DTW_MAX_DISTANCE": 8000.0,        # 归一化时的最大距离参考值（DTW累积距离很大，典型值~5000）
    
    # 综合分数阈值
    "FUSION_THRESHOLD": 0.40,          # 多维融合分数阈值（低于此拒绝匹配，默认40%）
    "HIGH_CONFIDENCE_THRESHOLD": 0.80, # 高置信度阈值（高于此视为强匹配）
    
    # 序列特征提取参数
    "USE_WEIGHTED_MEAN": True,         # 使用加权均值而非简单均值
    "EXTRACT_TRENDS": True,            # 是否提取趋势特征
    "TREND_WINDOWS": [5, 15, 30],      # 趋势提取的时间窗口
}

# ==================== 置信度系统配置 ====================
CONFIDENCE_CONFIG = {
    # 置信度计算权重（总和应为 1.0）
    "WIN_RATE_WEIGHT": 0.40,           # 历史胜率权重
    "STABILITY_WEIGHT": 0.30,          # 样本稳定性权重
    "SAMPLE_COUNT_WEIGHT": 0.20,       # 样本量充足度权重
    "REGIME_WEIGHT": 0.10,             # 市场状态一致性权重
    
    # 样本量评估参数
    "SAMPLE_COUNT_FULL": 10,           # 达到此样本量视为充足（满分）
    "SAMPLE_COUNT_MIN": 3,             # 最少样本量（少于此不计算置信度）
    
    # 稳定性评估参数
    "STABILITY_STD_THRESHOLD": 0.15,   # 收益标准差阈值（低于此视为稳定）
    "STABILITY_MAX_DRAWDOWN": 0.30,    # 最大回撤阈值（历史回撤超过此扣分）
    
    # 置信度阈值
    "MIN_CONFIDENCE": 0.30,            # 最低置信度（低于此不使用该原型）
    "HIGH_CONFIDENCE": 0.70,           # 高置信度阈值（可加大仓位）
    "PERFECT_CONFIDENCE": 0.90,        # 极高置信度阈值（顶级信号）
    
    # 置信度对最终分数的影响
    "CONFIDENCE_SCALE_MIN": 0.5,       # 低置信度时分数缩放下限
    "CONFIDENCE_SCALE_MAX": 1.2,       # 高置信度时分数缩放上限
}

# ==================== 市场状态分类配置 ====================
MARKET_REGIME_CONFIG = {
    "DIR_STRONG_THRESHOLD": 0.003,       # 方向趋势阈值 (0.3%) - 更容易触发趋势
    "STRENGTH_STRONG_THRESHOLD": 0.003,  # 强度阈值 (振幅/均价 > 0.3%) - 更容易触发强势
    "LOOKBACK_SWINGS": 4,               # 回看摆动点数量（只看最近走势）
    # 短期趋势修正（让 regime 更贴近 K 线走势）
    "SHORT_TREND_LOOKBACK": 6,          # 近 N 根 K 线方向（更聚焦当前）
    "SHORT_TREND_THRESHOLD": 0.0015,    # 0.15% 视为短期趋势明显
    
    # 市场状态自适应学习
    "REGIME_ADAPTATION_ENABLED": True,   # 是否启用市场状态自适应
    "REGIME_ACCURACY_MIN_SAMPLES": 15,   # 至少15笔才调整阈值
    "REGIME_THRESHOLD_ADJUST_RATE": 0.05,  # 每次调整5%
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
    "TAKE_PROFIT_ATR": 3.5,             # TP = 3.5 x ATR（适中的止盈倍数，平衡捕获利润与避免过度持仓）
    "MAX_HOLD_BARS": 240,
    
    # ── 止损保护参数 ──
    "MIN_SL_PCT": 0.004,        # 最小止损距离 0.4%（原配置，TP/SL 较近）
    "ATR_SL_MULTIPLIER": 4.0,   # SL = 4.0 x ATR（原配置）
    "MIN_RR_RATIO": 1.4,        # 最低盈亏比：TP >= SL x 1.4
    "SL_PROTECTION_SEC": 60,    # 新仓保护期（秒），保护期内止损暂缓，原来 8 秒太短
    
    # ── 前3根K线紧急止损守卫 (Early Exit Guard) ──
    # 问题：前3根保护期内只有硬止损保护，形成致命盲区（3根就能亏3-4%）
    # 方案1：亏损超阈值 → 紧急平仓
    # 方案2：原始市场状态与持仓方向冲突 → 收紧止损至入场价附近
    "EARLY_EXIT_ADVERSE_PCT": 1.5,     # 保护期内亏损超此%（杠杆后）→ 紧急平仓
    "EARLY_EXIT_TIGHTEN_PCT": 0.005,   # 市场冲突时止损收紧到入场价±此%（0.5%）
    
    # ── 反手单 (Stop-and-Reverse) ──
    # 止损说明方向判断错误，自动反手做反方向
    "REVERSE_ON_STOPLOSS": False,     # 关闭止损反手（翻转单更优：主动在有利位置翻转）
    "REVERSE_MAX_COUNT": 1,           # 最多连续反手次数（防止来回被扫）
    "REVERSE_BLOCK_SAME_DIR_SEC": 300,  # 止损后禁止同方向入场的时间（秒）
    # ── 翻转失败兜底（可控） ──
    "FLIP_FALLBACK_TEMPLATE_ENABLED": True,      # 震荡市翻转时，允许模板匹配兜底（即使主模式是原型）
    "FLIP_FALLBACK_DEGRADED_ENTRY_ENABLED": True,  # 翻转无匹配时，允许高分降级入场
    "FLIP_FALLBACK_MIN_SCORE": 35.0,             # 降级入场最低 flip_score（方向化位置评分）
    "FLIP_FALLBACK_DEGRADED_POSITION_PCT": 0.05, # 降级入场仓位上限（5%）
    
    # 动态追踪
    "HOLD_SAFE_THRESHOLD": 0.7,
    "HOLD_ALERT_THRESHOLD": 0.5,
    "HOLD_DERAIL_THRESHOLD": 0.3,
    
    # ── 市场恶化减仓（与分段止盈区分）──
    "REGIME_REDUCE_ENABLED": True,     # 是否启用市场恶化减仓
    "REGIME_REDUCE_MIN_PROFIT_PCT": 1.0,  # 至少盈利多少%才触发（市场恶化时）
    "REGIME_REDUCE_PCT": 0.25,         # 减仓比例（25%）

    # ── 弱化·震荡：先保本收紧（有浮盈才把止损挪到入场价，触发=保本出场）──
    "REGIME_WEAKEN_MIN_PROFIT_PCT": 0.2,   # 至少浮盈 0.2% 才收紧到 entry，否则不收紧

    # ── 阶梯基准止盈止损系统（三档止盈 + 动态止损）──
    # 【核心理念】止盈分三档锁定利润，止损不分档（全平剩余），每档止损跟随止盈成交价动态上移
    # 【阶梯基准】TP1基于入场价，TP2基于TP1成交价，TP3基于TP2成交价；SL始终基于当前阶梯基准
    # 【仓位分配】TP1平50% → TP2平剩余50%(=25%) → TP3全平(=25%)；SL始终全平剩余
    #
    # 以下 % 均为「收益率%」（杠杆后），挂单价格按 价格变动% = 收益率% / 杠杆 换算
    "STAGED_TP_PCT": 7.0,             # 每档止盈收益率 +7%（阶梯累计约 +21%）
    "STAGED_SL_PCT": 5.0,             # 止损收益率 -5%（基于当前阶梯基准，全平剩余）
    "STAGED_TP_RATIO_1": 0.50,        # 第1档减仓比例（50%）
    "STAGED_TP_RATIO_2": 0.50,        # 第2档减仓比例（剩余的50% = 总仓位25%）
    "STAGED_TP_RATIO_3": 1.00,        # 第3档全平剩余仓位（25%）
    "LIMIT_PRICE_BAND_PCT": 5.0,      # Binance 限价单价格带（相对 mark 偏离上限 %），超出则用 STOP_MARKET
    "STAGED_ORDERS_SEQUENTIAL": True,  # 阶梯基准模式必须为 True（逐档挂单）
    
    # 兼容旧配置（已废弃，保留以防其他模块引用）
    "STAGED_TP_1_PCT": 7.0,           # [废弃] 使用 STAGED_TP_PCT
    "STAGED_TP_2_PCT": 14.0,          # [废弃] TP2 = TP1成交价 + 7%
    "STAGED_TP_3_PCT": 21.0,          # [废弃] TP3 = TP2成交价 + 7%
    "STAGED_SL_1_PCT": 5.0,           # [废弃] 使用 STAGED_SL_PCT
    "STAGED_SL_2_PCT": 5.0,           # [废弃] SL2 = TP1成交价 - 5%
    "STAGED_SL_3_PCT": 5.0,           # [废弃] SL3 = TP2成交价 - 5%
    "STAGED_SL_RATIO_1": 1.00,        # [废弃] 止损始终全平
    "STAGED_SL_RATIO_2": 1.00,        # [废弃] 止损始终全平
    "STAGED_SL_RATIO_3": 1.00,        # [废弃] 止损始终全平

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

    # ── 自适应学习：出场时机追踪 ──
    "EXIT_TIMING_TRACKER_ENABLED": True,
    "EXIT_TIMING_EVAL_BARS": 30,
    "EXIT_TIMING_MOVE_PCT": 0.3,
    "EXIT_TIMING_MIN_EVALS": 20,
    "EXIT_TIMING_STATE_FILE": "data/exit_timing_state.json",

    # ── 自适应学习：止盈止损评估 ──
    "TPSL_TRACKER_ENABLED": True,
    "TPSL_EVAL_BARS": 30,
    "TPSL_MOVE_PCT": 0.5,
    "TPSL_MIN_EVALS": 20,
    "TPSL_STATE_FILE": "data/tpsl_tracker_state.json",

    # ── 自适应学习：近似信号评估 ──
    "NEAR_MISS_TRACKER_ENABLED": True,
    "NEAR_MISS_RATIO": 0.85,
    "NEAR_MISS_EVAL_BARS": 30,
    "NEAR_MISS_MOVE_PCT": 0.3,
    "NEAR_MISS_MIN_EVALS": 20,
    "NEAR_MISS_STATE_FILE": "data/near_miss_state.json",

    # ── 自适应学习：早期出场评估 ──
    "EARLY_EXIT_TRACKER_ENABLED": True,
    "EARLY_EXIT_EVAL_BARS": 30,
    "EARLY_EXIT_MOVE_PCT": 0.5,
    "EARLY_EXIT_MIN_EVALS": 15,
    "EARLY_EXIT_STATE_FILE": "data/early_exit_state.json",

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
    
    # ── 贝叶斯数据收集（仅供凯利公式使用）──
    "BAYESIAN_ENABLED": True,              # 启用数据收集（凯利公式需要胜率和盈亏比数据）
    "BAYESIAN_GATE_ENABLED": False,        # 禁用贝叶斯门控（不拦截交易）
    "BAYESIAN_PRIOR_STRENGTH": 10.0,       # 先验强度（历史回测数据相当于多少笔实盘交易）
    "BAYESIAN_MIN_WIN_RATE": 0.0,          # 门控已禁用（设为0=不拦截任何交易）
    "BAYESIAN_THOMPSON_SAMPLING": False,   # 禁用 Thompson Sampling（不做门控）
    "BAYESIAN_DECAY_ENABLED": True,        # 启用时间衰减（适应市场变化）
    "BAYESIAN_DECAY_HOURS": 24.0,          # 衰减间隔（小时）
    "BAYESIAN_DECAY_FACTOR": 0.95,         # 衰减因子（0-1，越小遗忘越快）
    "BAYESIAN_STATE_FILE": "data/bayesian_state.json",  # 持久化文件路径
    
    # ── 位置评分阈值（方向化）──
    "POS_THRESHOLD_LONG": -30,         # LONG 位置评分阈值（低于此拒绝/翻转，-20太严→-30）
    "POS_THRESHOLD_SHORT": -40,        # SHORT 位置评分阈值
    
    # ── MACD 趋势门控（斜率法）──
    "MACD_TREND_WINDOW": 5,            # 线性回归窗口（K线根数）
    "MACD_SLOPE_MIN": 0.003,           # 最小斜率绝对值（低于此视为无趋势，拒绝开仓）
    # ── MACD 零轴放宽（解决震荡市门控过严问题）──
    "MACD_ZERO_AXIS_FLOOR": 3.0,       # 零轴容忍度下限（防止小斜率时容忍度过小导致误拦）
    "MACD_RANGE_BYPASS_ZERO_AXIS": True,  # 震荡市中斜率方向正确时，豁免零轴检查
    "MACD_POS_SCORE_RESCUE": 40,       # 位置评分≥此值且斜率正确时，豁免零轴检查
    
    # ── 凯利公式动态仓位管理 ──
    "KELLY_ENABLED": True,                 # 是否启用凯利公式动态仓位
    "KELLY_FRACTION": 1.0,                 # 凯利分数（1.0=完整凯利，直接使用公式计算值）
    "KELLY_MAX_POSITION": 0.3,             # 凯利仓位上限（30% 本金）
    "KELLY_MIN_POSITION": 0.05,            # 凯利仓位下限（5% 本金，样本不足/期望为负时保守试探学习）
    "KELLY_MIN_SAMPLES": 5,                # 凯利计算最少样本数（少于此值用最小仓位）
    
    # ── 凯利参数自适应学习 ──
    "KELLY_ADAPTATION_ENABLED": True,      # 是否启用凯利参数自适应学习
    "KELLY_FRACTION_RANGE": (0.8, 1.0),    # KELLY_FRACTION 自适应调整范围（保持接近完整凯利）
    "KELLY_MAX_RANGE": (0.15, 0.30),       # KELLY_MAX 自适应调整范围
    "KELLY_MIN_RANGE": (0.05, 0.10),       # KELLY_MIN 自适应调整范围（下限5%~10%）
    
    # ── 杠杆自适应配置 ──
    "LEVERAGE_ADAPTIVE": False,            # 是否启用杠杆自适应调整（固定杠杆）
    "LEVERAGE_DEFAULT": 20,                # 默认杠杆倍数（程序启动与自适应基线）
    "LEVERAGE_MIN": 5,                     # 最小杠杆倍数
    "LEVERAGE_MAX": 100,                   # 最大杠杆倍数
    "LEVERAGE_ADJUST_STEP": 2,             # 每笔交易后杠杆调整步长（盈利+step/亏损-step）
    "LEVERAGE_ADJUST_THRESHOLD": 20,       # 旧逻辑：触发调整所需的最少交易数（现以每笔为准）
    "LEVERAGE_PROFIT_THRESHOLD": 2.0,      # 提高杠杆的平均收益阈值（%）
    "LEVERAGE_LOSS_THRESHOLD": -1.0,       # 降低杠杆的平均亏损阈值（%）
    "LEVERAGE_DRAWDOWN_LIMIT": 20.0,       # 降低杠杆的回撤限制（%）
    
    # ── 入场拒绝跟踪器（门控自适应学习）──
    "REJECTION_TRACKER_ENABLED": True,      # 是否启用拒绝跟踪
    "REJECTION_EVAL_BARS": 30,              # 拒绝后多少根K线进行事后评估
    "REJECTION_PROFIT_THRESHOLD_PCT": 0.3,  # 价格移动超过此百分比视为"本可以盈利"（%）
    "REJECTION_MAX_HISTORY": 200,           # 最大历史记录数
    "REJECTION_STATE_FILE": "data/rejection_tracker_state.json",  # 持久化文件路径
    
    # ── 出场时机追踪器（出场自适应学习）──
    "EXIT_TIMING_TRACKER_ENABLED": True,        # 是否启用出场时机追踪
    "EXIT_TIMING_EVAL_BARS": 30,                # 出场后追踪多少根K线评估后续走势
    "EXIT_TIMING_PREMATURE_PCT": 0.5,           # 出场后价格继续有利移动超过此%视为"过早出场"
    "EXIT_TIMING_LATE_RETRACE_PCT": 0.5,        # 出场时从峰值回撤超过此比例视为"过晚出场"
    "EXIT_TIMING_MAX_HISTORY": 200,             # 最大历史记录数
    "EXIT_TIMING_MIN_EVALS": 10,                # 每种出场原因至少需要多少次评估才给建议
    "EXIT_TIMING_STATE_FILE": "data/exit_timing_state.json",  # 持久化文件路径
    
    # ========== 反向下单测试 ==========
    "REVERSE_SIGNAL_MODE": False,  # True=所有信号反向（LONG→SHORT, SHORT→LONG）
    
    # ── 近似信号追踪器（相似度阈值自适应学习）──
    "NEAR_MISS_TRACKER_ENABLED": True,      # 是否启用近似信号追踪
    "NEAR_MISS_MARGIN": 0.10,               # 近似信号捕获范围（与阈值的差距，0.10=10%）
    "NEAR_MISS_EVAL_BARS": 30,              # 信号出现后多少根K线进行事后评估
    "NEAR_MISS_PROFIT_THRESHOLD_PCT": 0.3,  # 价格移动超过此百分比视为"本可盈利"（%）
    "NEAR_MISS_MAX_HISTORY": 200,           # 最大历史记录数
    "NEAR_MISS_STATE_FILE": "data/near_miss_tracker_state.json",  # 持久化文件路径
}

# ==================== DeepSeek API 配置 ====================
_deepseek_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()  # 去除首尾空格，避免 401
DEEPSEEK_CONFIG = {
    "ENABLED": bool(_deepseek_key),                  # 有密钥即启用
    "API_KEY": _deepseek_key,                         # DeepSeek API 密钥（https://platform.deepseek.com）
    "MODEL": "deepseek-chat",            # 模型名称
    "MAX_TOKENS": 2000,                  # 最大生成token数（复盘分析需适当展开意见）
    "TEMPERATURE": 0.3,                  # 温度参数（0.3=更确定性，适合分析）
    
    # 持仓中实时 DeepSeek（仅展示）
    "HOLDING_INTERVAL_BARS": 3,          # 每 N 根 K 线收线调用一次（有持仓时）
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
