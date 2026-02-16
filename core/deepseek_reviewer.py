"""
R3000 DeepSeek AI 交易复盘分析器
在每笔交易结束后，异步调用 DeepSeek API 进行深度分析

功能：
  - 分析入场质量（时机、指标确认、市场状态）
  - 识别关键转折点（何时应该加仓/减仓/离场）
  - 评估出场质量（过早/过晚/合理）
  - 检查市场状态分类准确性
  - 提供具体的参数调整建议

输出：
  - 结构化的中文分析报告
  - 存储在 data/deepseek_reviews/ 目录
  - UI展示在自适应学习Tab的"AI复盘"栏
"""

import json
import os
import time
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import threading

from core.paper_trader import PaperOrder, OrderSide, CloseReason


@dataclass
class TradeContext:
    """交易上下文（发送给DeepSeek的数据）"""
    # 基本信息
    order_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    hold_bars: int
    profit_pct: float
    close_reason: str
    
    # 入场状态
    entry_snapshot: Dict
    entry_reason: str
    entry_similarity: float
    market_regime_at_entry: str
    
    # 出场状态
    exit_snapshot: Optional[Dict]
    peak_profit_pct: float
    
    # 持仓过程（开仓→平仓之间的指标快照）
    indicator_snapshots_during_hold: Optional[List[Dict]] = None
    
    # 仓位大小
    position_pct: float = 0.5  # 仓位占账户的百分比
    
    # 原型历史表现
    prototype_stats: Optional[Dict] = None
    
    # 反事实分析结果
    counterfactual_result: Optional[Dict] = None
    
    # 特征模式统计
    feature_patterns: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "hold_bars": self.hold_bars,
            "profit_pct": self.profit_pct,
            "close_reason": self.close_reason,
            "entry_snapshot": self.entry_snapshot,
            "entry_reason": self.entry_reason,
            "entry_similarity": self.entry_similarity,
            "market_regime_at_entry": self.market_regime_at_entry,
            "exit_snapshot": self.exit_snapshot,
            "peak_profit_pct": self.peak_profit_pct,
            "position_pct": self.position_pct,
            "prototype_stats": self.prototype_stats,
            "counterfactual_result": self.counterfactual_result,
            "feature_patterns": self.feature_patterns,
        }
    
    @staticmethod
    def from_order(order: PaperOrder, 
                   counterfactual_result: Optional[Dict] = None,
                   prototype_stats: Optional[Dict] = None,
                   feature_patterns: Optional[Dict] = None,
                   position_pct: float = 0.5) -> 'TradeContext':
        """从PaperOrder创建TradeContext"""
        entry_snapshot = {}
        if hasattr(order, 'entry_snapshot') and order.entry_snapshot:
            if hasattr(order.entry_snapshot, 'to_dict'):
                entry_snapshot = order.entry_snapshot.to_dict()
            elif isinstance(order.entry_snapshot, dict):
                entry_snapshot = order.entry_snapshot
        
        exit_snapshot = {}
        if hasattr(order, 'exit_snapshot') and order.exit_snapshot:
            if hasattr(order.exit_snapshot, 'to_dict'):
                exit_snapshot = order.exit_snapshot.to_dict()
            elif isinstance(order.exit_snapshot, dict):
                exit_snapshot = order.exit_snapshot
        
        snapshots = getattr(order, 'indicator_snapshots', None) or getattr(order, 'indicator_snapshots_during_hold', None) or []
        
        return TradeContext(
            order_id=order.order_id,
            symbol=order.symbol,
            direction=order.side.value,
            entry_price=order.entry_price,
            exit_price=order.exit_price or 0.0,
            entry_time=order.entry_time.isoformat() if order.entry_time else "",
            exit_time=order.exit_time.isoformat() if order.exit_time else "",
            hold_bars=order.hold_bars,
            profit_pct=order.profit_pct,
            close_reason=order.close_reason.value if order.close_reason else "未知",
            entry_snapshot=entry_snapshot,
            entry_reason=order.entry_reason,
            entry_similarity=order.entry_similarity,
            market_regime_at_entry=getattr(order, 'regime_at_entry', '未知'),
            exit_snapshot=exit_snapshot,
            peak_profit_pct=order.peak_profit_pct,
            indicator_snapshots_during_hold=snapshots if isinstance(snapshots, list) else list(snapshots) if snapshots else None,
            position_pct=position_pct,
            prototype_stats=prototype_stats,
            counterfactual_result=counterfactual_result,
            feature_patterns=feature_patterns,
        )


class DeepSeekReviewer:
    """
    DeepSeek AI 交易复盘分析器
    异步调用API，不阻塞主线程
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: DEEPSEEK_CONFIG from config.py
        """
        self.enabled = config.get("ENABLED", False)
        self.api_key = (config.get("API_KEY", "") or "").strip()
        self.model = config.get("MODEL", "deepseek-chat")
        self.max_tokens = config.get("MAX_TOKENS", 2000)
        self.temperature = config.get("TEMPERATURE", 0.3)
        # 官方示例使用 /chat/completions（无 /v1），与 OpenAI 兼容的 v1 路径若返回 401 可尝试此处
        self.api_url = "https://api.deepseek.com/chat/completions"
        
        # 结果存储目录
        self.reviews_dir = "data/deepseek_reviews"
        os.makedirs(self.reviews_dir, exist_ok=True)

        if self.enabled and self.api_key:
            print(f"[DeepSeekReviewer] API Key 已加载 (长度 {len(self.api_key)} 字符)，端点: {self.api_url}")
        elif self.enabled and not self.api_key:
            print("[DeepSeekReviewer] 已启用但未读到 API Key，请检查 .env 中 DEEPSEEK_API_KEY 与启动目录")

        # 异步任务队列
        self.pending_reviews: List[TradeContext] = []
        self.review_thread: Optional[threading.Thread] = None
        self.running = False
    
    def start_background_worker(self):
        """启动后台工作线程"""
        if self.running or not self.enabled:
            return
        
        self.running = True
        self.review_thread = threading.Thread(target=self._review_worker, daemon=True)
        self.review_thread.start()
        print("[DeepSeekReviewer] 后台工作线程已启动")
    
    def stop_background_worker(self):
        """停止后台工作线程"""
        self.running = False
        if self.review_thread and self.review_thread.is_alive():
            self.review_thread.join(timeout=5)
        print("[DeepSeekReviewer] 后台工作线程已停止")
    
    def add_trade_for_review(self, trade_context: TradeContext):
        """
        添加一笔交易到待复盘队列
        
        Args:
            trade_context: 交易上下文
        """
        if not self.enabled:
            return
        
        self.pending_reviews.append(trade_context)
        print(f"[DeepSeekReviewer] 已添加交易 {trade_context.order_id} 到复盘队列")
    
    def _review_worker(self):
        """后台工作线程，处理待复盘的交易"""
        while self.running:
            if not self.pending_reviews:
                time.sleep(2)
                continue
            
            # 取出一笔交易
            trade_context = self.pending_reviews.pop(0)
            
            try:
                # 调用API分析
                review_result = self._review_trade_sync(trade_context)
                
                # 保存结果
                self._save_review(trade_context.order_id, review_result)
                
                print(f"[DeepSeekReviewer] 交易 {trade_context.order_id} 复盘完成")
            except Exception as e:
                print(f"[DeepSeekReviewer] 复盘失败 {trade_context.order_id}: {e}")
            
            # 限速：每次请求间隔2秒
            time.sleep(2)
    
    def _review_trade_sync(self, trade_context: TradeContext) -> Dict:
        """
        同步调用DeepSeek API分析交易
        
        Returns:
            {
                "analysis": "AI分析文本",
                "timestamp": "...",
                "model": "deepseek-chat",
                "error": None or "错误信息"
            }
        """
        if not self.api_key:
            return {
                "analysis": "未配置DeepSeek API Key",
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "error": "NO_API_KEY",
            }
        
        # 构建提示词
        prompt = self._build_prompt(trade_context)
        
        # 调用API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一位专业的量化交易分析师，擅长复盘交易、识别问题、提出改进建议。请用中文回答，对每一条检查项给出明确结论：通过则简要说明理由，有问题则展开分析原因、可能后果、以及具体改进建议（含数值或逻辑）。不要只写结论，关键项要有 1～3 句论证或建议。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                return {
                    "analysis": analysis_text,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "error": None,
                    "trade_context": trade_context.to_dict(),
                }
            else:
                error_msg = f"API返回错误: {response.status_code} {response.text}"
                return {
                    "analysis": "",
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "error": error_msg,
                }
        
        except Exception as e:
            return {
                "analysis": "",
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "error": str(e),
            }
    
    def _build_prompt(self, ctx: TradeContext) -> str:
        """构建发送给DeepSeek的提示词"""
        
        # 基本信息
        prompt = f"""请分析以下交易的表现并提供改进建议：

## 交易概况
- 订单ID: {ctx.order_id}
- 方向: {ctx.direction}
- 入场价: {ctx.entry_price:.2f}
- 出场价: {ctx.exit_price:.2f}
- 持仓时长: {ctx.hold_bars} 根K线
- 收益率: {ctx.profit_pct:+.2f}%
- 峰值收益: {ctx.peak_profit_pct:+.2f}%
- 平仓原因: {ctx.close_reason}

## 入场状态
- 市场状态: {ctx.market_regime_at_entry}
- 相似度: {ctx.entry_similarity:.2%}
- 入场理由: {ctx.entry_reason}
"""
        
        # 入场指标快照（开仓时刻）
        if ctx.entry_snapshot:
            prompt += f"""
## 入场时指标（开仓时刻）
- KDJ: J={ctx.entry_snapshot.get('kdj_j', 0):.1f}, D={ctx.entry_snapshot.get('kdj_d', 0):.1f}, K={ctx.entry_snapshot.get('kdj_k', 0):.1f} (趋势: {ctx.entry_snapshot.get('kdj_trend', 'flat')})
- MACD: 柱={ctx.entry_snapshot.get('macd_hist', 0):.2f}, 斜率={ctx.entry_snapshot.get('macd_hist_slope', 0):.2f}
- RSI: {ctx.entry_snapshot.get('rsi', 50):.1f} | ADX: {ctx.entry_snapshot.get('adx', 0):.1f}
- 布林位置: {ctx.entry_snapshot.get('boll_position', 0.5):.2f} (0=下轨 1=上轨) | ATR: {ctx.entry_snapshot.get('atr', 0):.2f}
"""
        
        # 出场指标快照（平仓时刻）
        if ctx.exit_snapshot:
            prompt += f"""
## 出场时指标（平仓时刻）
- KDJ: J={ctx.exit_snapshot.get('kdj_j', 0):.1f}, D={ctx.exit_snapshot.get('kdj_d', 0):.1f}, K={ctx.exit_snapshot.get('kdj_k', 0):.1f} (趋势: {ctx.exit_snapshot.get('kdj_trend', 'flat')})
- MACD: 柱={ctx.exit_snapshot.get('macd_hist', 0):.2f}, 斜率={ctx.exit_snapshot.get('macd_hist_slope', 0):.2f}
- RSI: {ctx.exit_snapshot.get('rsi', 50):.1f} | ADX: {ctx.exit_snapshot.get('adx', 0):.1f}
- 布林位置: {ctx.exit_snapshot.get('boll_position', 0.5):.2f} | ATR: {ctx.exit_snapshot.get('atr', 0):.2f}
"""
        
        # 持仓过程（开仓→平仓的指标变化）
        snaps = ctx.indicator_snapshots_during_hold or []
        if snaps:
            recent = snaps[-8:] if len(snaps) >= 8 else snaps
            prompt += "\n## 持仓过程（指标变化）\n"
            for i, s in enumerate(recent):
                if isinstance(s, dict):
                    bar = s.get('bar_idx', i)
                    kdj = s.get('kdj_j', 0)
                    macd = s.get('macd_hist', 0)
                    rsi = s.get('rsi', 50)
                    price = s.get('price', 0)
                    prompt += f"- 第{bar}根: 价={price:.2f} KDJ-J={kdj:.0f} MACD={macd:+.1f} RSI={rsi:.0f}\n"
        
        # 反事实分析
        if ctx.counterfactual_result:
            cf = ctx.counterfactual_result
            prompt += "\n## 反事实分析（如果当时这样做会怎样？）\n"
            
            if cf.get('entry_conclusion'):
                prompt += f"- 入场时机: {cf['entry_conclusion']}\n"
            
            if cf.get('exit_conclusion'):
                prompt += f"- 出场时机: {cf['exit_conclusion']}\n"
            
            if cf.get('tpsl_conclusion'):
                prompt += f"- 止盈止损: {cf['tpsl_conclusion']}\n"
        
        # 原型历史表现
        if ctx.prototype_stats:
            stats = ctx.prototype_stats
            prompt += f"""
## 原型历史表现
- 匹配次数: {stats.get('match_count', 0)}
- 胜率: {stats.get('win_rate', 0):.1%}
- 平均收益: {stats.get('avg_profit', 0):+.2f}%
"""
        
        # 特征模式统计
        if ctx.feature_patterns:
            prompt += "\n## 特征模式学习\n"
            for feature, (p25, p75) in ctx.feature_patterns.items():
                prompt += f"- {feature} 盈利范围: [{p25:.2f}, {p75:.2f}]\n"
        
        # 分析任务 - 增强版：遍历所有入场/持仓/出场条件
        prompt += """

请从以下角度**逐条遍历分析**并给出专业建议：

## 分析框架（请逐项检查）

### 1. 入场条件评估（逐一检查）
请逐条评估以下入场条件，指出是否合理：

- [ ] **相似度阈值**：{ctx.entry_similarity:.2%}是否足够可信？是否应该更严格或放宽？
- [ ] **MACD确认**：MACD柱状图={ctx.entry_snapshot.get('macd_hist', 0):.2f}, 斜率={ctx.entry_snapshot.get('macd_hist_slope', 0):.2f}，是否充分确认趋势？
- [ ] **KDJ位置**：J={ctx.entry_snapshot.get('kdj_j', 0):.1f}, D={ctx.entry_snapshot.get('kdj_d', 0):.1f}, K={ctx.entry_snapshot.get('kdj_k', 0):.1f}，是否在理想区间？趋势={ctx.entry_snapshot.get('kdj_trend', 'flat')}是否支持方向？
- [ ] **RSI过滤**：RSI={ctx.entry_snapshot.get('rsi', 50):.1f}，是否避免了超买超卖区域？
- [ ] **市场状态分类**："{ctx.market_regime_at_entry}"分类是否准确？是否与实际价格走势一致？
- [ ] **ADX趋势强度**：ADX={ctx.entry_snapshot.get('adx', 0):.1f}，趋势是否足够强劲？
- [ ] **布林带位置**：位置={ctx.entry_snapshot.get('boll_position', 0.5):.2f} (0=下轨, 1=上轨)，是否合理？
- [ ] **入场时机**：是否应该再等1-2根K线以获得更好的确认？

### 2. 持仓管理评估（关键转折点）
请检查持仓期间的关键信号和管理决策：

- [ ] **止盈距离**：峰值利润{ctx.peak_profit_pct:+.2f}%，实际收益{ctx.profit_pct:+.2f}%，止盈距离是否合理？
- [ ] **止损距离**：如果亏损，止损距离是否过大或过小？
- [ ] **峰值回撤**：从峰值{ctx.peak_profit_pct:+.2f}%回撤到{ctx.profit_pct:+.2f}%，回撤{ctx.peak_profit_pct - ctx.profit_pct:.1f}%，是否错过了止盈时机？
- [ ] **反转信号识别**：回撤过程中，是否出现了明显的反转信号（KDJ死叉、MACD转负、RSI背离）被忽略？
- [ ] **减仓时机**：是否有提前减仓的机会（布林带极值、RSI极端、动能衰减）？
- [ ] **追踪止盈**：追踪止盈策略是否及时激活？参数是否过于保守或激进？

### 3. 出场决策评估（时机与原因）
请评估出场质量和时机：

- [ ] **出场原因合理性**：平仓原因"{ctx.close_reason}"是否合理？是主动止盈/止损还是被动触发？
- [ ] **出场时机早晚**：是否出场过早？（后续价格是否继续沿预期方向运动？）是否出场过晚？（是否应该在峰值附近离场？）
- [ ] **离场信号确认**：出场时是否有明确的技术信号支持？还是仅依赖固定参数？
- [ ] **追踪止盈效果**：如果使用了追踪止盈，效果如何？是否锁定了大部分利润？

### 4. 市场状态准确性（事后验证）
请验证市场状态判断是否准确：

- [ ] **入场市场状态验证**：入场时判断为"{ctx.market_regime_at_entry}"，实际价格走势是否验证了这个判断？
- [ ] **持仓期间状态转变**：持仓{ctx.hold_bars}根K线期间，市场状态是否发生转变？
- [ ] **状态与策略匹配**：如果市场状态发生变化，是否应该调整持仓策略？
- [ ] **方向与状态一致性**：{ctx.direction}方向与市场状态"{ctx.market_regime_at_entry}"是否匹配？

### 5. 凯利仓位合理性（如果适用）
如果有仓位信息，请评估：

- [ ] **仓位大小**：本次仓位是否合理？基于收益结果，是否应该加大或减小？
- [ ] **凯利分数调整**：是否建议调整凯利分数（KELLY_FRACTION）？

### 6. 具体参数调整建议
请给出可操作的具体建议（必须包含数值）：

- [ ] **相似度阈值**：建议值=？（当前系统阈值约0.65）
- [ ] **止盈距离**：建议值=？ATR 或 %（当前约3.0 ATR）
- [ ] **止损距离**：建议值=？ATR 或 %（当前约1.5 ATR）
- [ ] **追踪止盈参数**：激活阈值=？%，追踪距离=？%
- [ ] **MACD斜率阈值**：建议值=？（用于入场确认）
- [ ] **KDJ过滤条件**：J值范围=？（例如：20-80）
- [ ] **其他过滤条件**：建议增加哪些额外过滤？

## 重要说明
⚠️ 你的分析**仅供参考**，不会自动修改系统参数。
✓ 对**有问题的项**：请展开写清原因、影响、以及可操作改进（含数值或逻辑），每条至少 1～3 句。
✓ 对**通过的项**：可简要写"✓ 合理"并附一句理由。
✓ 重点突出**实际问题**和**具体改进方向**，不必刻意压缩字数，保证意见充分、可执行。
"""
        
        return prompt
    
    def request_holding_advice_async(self, order, df, state, reasoning_result, on_result):
        """
        持仓中实时 DeepSeek 建议（交易员角色）
        
        传入：持仓摘要、5层推理、指标、系统决策
        返回：是否继续持仓/是否平仓 + 对系统决策的评判
        先仅展示，不参与执行。
        
        Args:
            order: PaperOrder 当前持仓
            df: DataFrame K线数据
            state: EngineState（用于写入结果）
            reasoning_result: ReasoningResult 5层推理
            on_result: callback(state) 结果回调，在主线程更新 state
        """
        if not self.enabled or not self.api_key:
            return
        def _worker():
            try:
                prompt = self._build_holding_prompt(order, df, state, reasoning_result)
                result = self._call_holding_api(prompt)
                if result and on_result:
                    on_result(result)
            except Exception as e:
                if on_result:
                    on_result({"advice": "", "judgement": f"请求失败: {e}", "error": str(e)})
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
    
    def _build_holding_prompt(self, order, df, state, reasoning_result) -> str:
        """构建持仓中 DeepSeek 提示词（含开仓、持仓过程、当前指标）"""
        side = getattr(order, 'side', None)
        side_str = side.value if hasattr(side, 'value') else "LONG"
        entry_price = getattr(order, 'entry_price', 0)
        current_price = state.current_price if hasattr(state, 'current_price') else entry_price
        profit_pct = getattr(order, 'profit_pct', 0)
        peak_pct = getattr(order, 'peak_profit_pct', 0)
        similarity = getattr(order, 'current_similarity', 0) or getattr(order, 'entry_similarity', 0)
        hold_bars = getattr(order, 'hold_bars', 0)
        regime = getattr(state, 'market_regime', '未知')
        regime_at_entry = getattr(order, 'regime_at_entry', regime)
        exit_sug = getattr(state, 'holding_exit_suggestion', '继续持有')
        tpsl_action = getattr(state, 'tpsl_action', 'hold')
        position_sug = getattr(state, 'position_suggestion', '维持')
        tp = getattr(order, 'take_profit', None)
        sl = getattr(order, 'stop_loss', None)
        partial_tp = getattr(order, 'partial_tp_count', 0)
        partial_sl = getattr(order, 'partial_sl_count', 0)
        partial_tp_count = getattr(order, 'partial_tp_count', 0)
        entry_atr = getattr(order, 'entry_atr', 0)
        
        # 当前K线指标（从 df 最后一根读取）
        curr = {}
        if df is not None and len(df) > 0:
            row = df.iloc[-1]
            curr = {
                'kdj_j': row.get('kdj_j', row.get('j', 50)),
                'kdj_d': row.get('kdj_d', row.get('d', 50)),
                'kdj_k': row.get('kdj_k', row.get('k', 50)),
                'macd_hist': row.get('macd_hist', 0),
                'macd_signal': row.get('macd_signal', 0),
                'rsi': row.get('rsi_14', row.get('rsi', 50)),
                'adx': row.get('adx', 25),
                'boll_position': row.get('boll_position', 0.5),
                'atr': row.get('atr', 0),
                'volume_ratio': row.get('volume_ratio', 1.0),
                'open': row.get('open', current_price),
                'high': row.get('high', current_price),
                'low': row.get('low', current_price),
                'close': row.get('close', current_price),
            }
        
        # 入场时指标（entry_snapshot 或 indicator_snapshots 首根）
        entry_ind = {}
        if hasattr(order, 'entry_snapshot') and order.entry_snapshot:
            sn = order.entry_snapshot
            if hasattr(sn, 'to_dict'):
                entry_ind = sn.to_dict()
            elif isinstance(sn, dict):
                entry_ind = sn
        snaps = getattr(order, 'indicator_snapshots', []) or getattr(order, 'indicator_snapshots_during_hold', [])
        if not entry_ind and snaps:
            entry_ind = snaps[0] if isinstance(snaps[0], dict) else {}
        
        prompt = f"""你是一位专业期货交易员，正在管理一笔持仓。请基于**开仓、持仓过程、当前指标**的全过程数据给出决策建议，并评判当前系统决策是否合理。

## 一、开仓与持仓摘要
- 方向: {side_str}
- 入场价: {entry_price:.2f} | 当前价: {current_price:.2f}
- 盈亏(含杠杆): {profit_pct:+.2f}% | 峰值: {peak_pct:+.2f}%
- 止盈价: {tp} | 止损价: {sl}
- 相似度: {similarity:.1%} | 持仓: {hold_bars}根K线
- 分段止盈: {partial_tp}次 | 分段止损: {partial_sl}次
- 入场ATR: {entry_atr:.2f} | 市场(入场→当前): {regime_at_entry} → {regime}

## 二、入场时指标（开仓时刻）
"""
        if entry_ind:
            prompt += f"""- KDJ: J={entry_ind.get('kdj_j', 0):.1f} D={entry_ind.get('kdj_d', 0):.1f} K={entry_ind.get('kdj_k', 0):.1f} 趋势={entry_ind.get('kdj_trend', 'flat')}
- MACD: 柱={entry_ind.get('macd_hist', 0):.2f} 斜率={entry_ind.get('macd_hist_slope', 0):.2f}
- RSI: {entry_ind.get('rsi', 50):.1f} | ADX: {entry_ind.get('adx', 25):.1f}
- 布林位置: {entry_ind.get('boll_position', 0.5):.2f} (0下轨 1上轨) | ATR: {entry_ind.get('atr', 0):.2f}
"""
        else:
            prompt += "- (无快照)\n"
        
        prompt += f"""
## 三、当前K线指标（最新）
- KDJ: J={curr.get('kdj_j', 50):.1f} D={curr.get('kdj_d', 50):.1f} K={curr.get('kdj_k', 50):.1f}
- MACD: 柱={curr.get('macd_hist', 0):.2f} 信号={curr.get('macd_signal', 0):.2f}
- RSI: {curr.get('rsi', 50):.1f} | ADX: {curr.get('adx', 25):.1f}
- 布林位置: {curr.get('boll_position', 0.5):.2f} (0下轨 1上轨)
- ATR: {curr.get('atr', 0):.2f} | 量比: {curr.get('volume_ratio', 1.0):.2f}
- 价格: O={curr.get('open', curr.get('close', current_price)):.2f} H={curr.get('high', current_price):.2f} L={curr.get('low', current_price):.2f} C={curr.get('close', current_price):.2f}

## 四、持仓过程（最近指标变化）
"""
        # 持仓期间指标快照摘要（indicator_snapshots 每根K线存一次）
        snaps = getattr(order, 'indicator_snapshots', []) or getattr(order, 'indicator_snapshots_during_hold', [])
        if snaps:
            recent = snaps[-5:] if len(snaps) >= 5 else snaps
            for i, s in enumerate(recent):
                if isinstance(s, dict):
                    bar = s.get('bar_idx', i)
                    kdj = s.get('kdj_j', 0)
                    macd = s.get('macd_hist', 0)
                    rsi = s.get('rsi', 50)
                    prompt += f"- 第{bar}根: KDJ-J={kdj:.0f} MACD柱={macd:+.1f} RSI={rsi:.0f}\n"
        else:
            prompt += "- (无过程快照)\n"
        
        prompt += """
## 五、5层推理与系统决策
"""
        if reasoning_result and hasattr(reasoning_result, 'layers'):
            for i, layer in enumerate(reasoning_result.layers[:5]):
                prompt += f"- {layer.name}: {layer.summary}\n"
        if reasoning_result and hasattr(reasoning_result, 'verdict'):
            prompt += f"- 综合判决: {reasoning_result.verdict}\n"
        if reasoning_result and hasattr(reasoning_result, 'narrative'):
            prompt += f"- 综合叙述: {reasoning_result.narrative}\n"
        
        prompt += f"""
## 当前系统决策
- 止盈建议: {exit_sug}
- TP/SL动作: {tpsl_action}
- 仓位建议: {position_sug}

请用两段回答：
1. **决策**：是否建议继续持仓？是否建议平仓或部分止盈？1-2句理由。
2. **评判**：对当前系统决策的评判（同意/更保守/更激进 + 简短理由）。
"""
        return prompt
    
    def _call_holding_api(self, prompt: str) -> Optional[Dict]:
        """调用 DeepSeek API 获取持仓建议"""
        if not self.api_key:
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一位专业期货交易员，会收到开仓、持仓过程、当前指标的全过程数据。请基于这些指标（KDJ/MACD/RSI/ADX/布林/ATR等）和技术逻辑，结合开仓→持仓→当前的变化趋势，给出决策建议。用中文回答，简洁专业。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 400,
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"advice": content, "judgement": "", "error": None}
            if response.status_code == 401:
                try:
                    err_body = response.json()
                    err_msg = err_body.get("error", {}).get("message", response.text) if isinstance(err_body, dict) else response.text
                except Exception:
                    err_msg = response.text
                print(f"[DeepSeekReviewer] 401 响应: {err_msg[:200]}")
                return {
                    "advice": "",
                    "judgement": f"API 认证失败(401)：{err_msg[:80]}",
                    "error": response.text,
                }
            return {"advice": "", "judgement": f"API错误: {response.status_code}", "error": response.text}
        except Exception as e:
            return {"advice": "", "judgement": f"请求失败: {e}", "error": str(e)}
    
    def request_idle_advice_async(self, state, df, on_result):
        """
        无持仓时每 2 分钟请求一次：当前市场/等待建议（1～2 句）。
        on_result(result_dict) 在后台线程调用，主线程需通过信号接收。
        """
        if not self.enabled or not self.api_key:
            return
        price = getattr(state, "current_price", 0) or 0
        regime = getattr(state, "market_regime", "未知")
        bars = getattr(state, "total_bars", 0)
        recent = ""
        if df is not None and len(df) >= 3:
            try:
                closes = df["close"].iloc[-5:].tolist()
                recent = "最近5根收盘: " + ", ".join(f"{c:.0f}" for c in closes)
            except Exception:
                recent = ""
        prompt = f"""当前无持仓，等待信号。请给 1～2 句简要意见（市场状态、是否适合观望或可关注方向）。
- 当前价: {price:.2f}
- 市场状态: {regime}
- 已处理K线: {bars}
{("- " + recent) if recent else ""}
请用中文回答，1～2 句即可。"""
        def _worker():
            try:
                result = self._call_holding_api(prompt)
                if result and on_result:
                    on_result(result)
            except Exception as e:
                if on_result:
                    on_result({"advice": "", "judgement": f"请求失败: {e}", "error": str(e)})
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
    
    def _save_review(self, order_id: str, review_result: Dict):
        """保存复盘结果到文件"""
        filename = f"{order_id}.json"
        filepath = os.path.join(self.reviews_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(review_result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DeepSeekReviewer] 保存复盘结果失败: {e}")
    
    def load_review(self, order_id: str) -> Optional[Dict]:
        """加载某笔交易的复盘结果"""
        filename = f"{order_id}.json"
        filepath = os.path.join(self.reviews_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[DeepSeekReviewer] 加载复盘结果失败: {e}")
            return None
    
    def get_all_reviews(self, limit: int = 20) -> List[Dict]:
        """获取最近的复盘结果"""
        reviews = []
        
        if not os.path.exists(self.reviews_dir):
            return reviews
        
        files = sorted(
            [f for f in os.listdir(self.reviews_dir) if f.endswith('.json')],
            key=lambda x: os.path.getmtime(os.path.join(self.reviews_dir, x)),
            reverse=True
        )
        
        for filename in files[:limit]:
            filepath = os.path.join(self.reviews_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    reviews.append(json.load(f))
            except:
                continue
        
        return reviews


# 简单测试
if __name__ == "__main__":
    # 模拟配置
    test_config = {
        "ENABLED": False,  # 测试时禁用真实API调用
        "API_KEY": "sk-xxx",
        "MODEL": "deepseek-chat",
        "MAX_TOKENS": 800,
        "TEMPERATURE": 0.3,
    }
    
    reviewer = DeepSeekReviewer(test_config)
    
    # 模拟交易上下文
    test_context = TradeContext(
        order_id="TEST_001",
        symbol="BTCUSDT",
        direction="LONG",
        entry_price=65000.0,
        exit_price=65300.0,
        entry_time=datetime.now().isoformat(),
        exit_time=datetime.now().isoformat(),
        hold_bars=45,
        profit_pct=4.6,
        close_reason="止盈",
        entry_snapshot={
            "kdj_j": 72.5,
            "kdj_d": 65.3,
            "kdj_k": 68.1,
            "kdj_trend": "rising",
            "macd_hist": 15.2,
            "macd_hist_slope": 2.3,
            "rsi": 58.5,
            "adx": 28.5,
            "boll_position": 0.65,
        },
        entry_reason="原型匹配：LONG_Proto_05",
        entry_similarity=0.82,
        market_regime_at_entry="bullish_strong",
        exit_snapshot={},
        peak_profit_pct=5.8,
    )
    
    print("DeepSeekReviewer 模块加载成功！")
    print(f"配置: {test_config}")
