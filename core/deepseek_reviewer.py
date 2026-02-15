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
        self.api_key = config.get("API_KEY", "")
        self.model = config.get("MODEL", "deepseek-chat")
        self.max_tokens = config.get("MAX_TOKENS", 800)
        self.temperature = config.get("TEMPERATURE", 0.3)
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # 结果存储目录
        self.reviews_dir = "data/deepseek_reviews"
        os.makedirs(self.reviews_dir, exist_ok=True)
        
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
                    "content": "你是一位专业的量化交易分析师，擅长复盘交易、识别问题、提出改进建议。请用中文回答，简洁且有针对性。"
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
        
        # 入场指标快照
        if ctx.entry_snapshot:
            prompt += f"""
- KDJ: J={ctx.entry_snapshot.get('kdj_j', 0):.1f}, D={ctx.entry_snapshot.get('kdj_d', 0):.1f}, K={ctx.entry_snapshot.get('kdj_k', 0):.1f} (趋势: {ctx.entry_snapshot.get('kdj_trend', 'flat')})
- MACD: 柱状图={ctx.entry_snapshot.get('macd_hist', 0):.2f}, 斜率={ctx.entry_snapshot.get('macd_hist_slope', 0):.2f}
- RSI: {ctx.entry_snapshot.get('rsi', 50):.1f}
- ADX: {ctx.entry_snapshot.get('adx', 0):.1f}
- 布林位置: {ctx.entry_snapshot.get('boll_position', 0.5):.2f} (0=下轨, 1=上轨)
"""
        
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
✓ 请给出**具体的、可操作的**建议，包含数值和逻辑。
✓ 如果某项检查通过，简单说"✓ 合理"；如果有问题，详细说明**为什么**有问题以及**如何改进**。
✓ 重点突出**实际问题**和**具体改进方向**。

请简洁回答，控制在800字以内。
"""
        
        return prompt
    
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
