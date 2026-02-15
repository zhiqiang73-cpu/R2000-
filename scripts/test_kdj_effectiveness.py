"""
KDJ金叉胜率回测脚本
测试：如果在KDJ金叉时入场，实际胜率如何？
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_kdj(df, period=9, smooth_k=3, smooth_d=3):
    """计算KDJ指标"""
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)
    
    k = rsv.ewm(span=smooth_k, adjust=False).mean()
    d = k.ewm(span=smooth_d, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def test_kdj_golden_cross_strategy():
    """测试KDJ金叉策略的实际胜率"""
    
    # 1. 加载数据
    data_file = Path(__file__).parent.parent / "btcusdt_1m.parquet"
    if not data_file.exists():
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    print("📊 加载数据...")
    df = pd.read_parquet(data_file)
    
    # 采样最近10万根K线（避免全量计算）
    if len(df) > 100000:
        df = df.iloc[-100000:].copy()
    
    print(f"✅ 数据加载完成: {len(df)} 根K线")
    
    # 2. 计算KDJ指标
    print("📈 计算KDJ指标...")
    df['k'], df['d'], df['j'] = calculate_kdj(df)
    
    # 3. 识别KDJ金叉信号
    # 金叉条件：当前K线 J上穿D（J[-1] < D[-1] and J[-0] > D[-0]）
    df['j_cross_d'] = (
        (df['j'].shift(1) < df['d'].shift(1)) &  # 前一根 J < D
        (df['j'] > df['d'])                       # 当前根 J > D
    )
    
    # 4. 低位金叉（J < 20）
    df['low_golden_cross'] = df['j_cross_d'] & (df['j'] < 20)
    
    # 5. 模拟交易
    results = {
        'all_golden_cross': [],    # 所有金叉
        'low_golden_cross': [],    # 低位金叉（J < 20）
    }
    
    for signal_type in ['all_golden_cross', 'low_golden_cross']:
        signals = df[df[signal_type]].index
        
        print(f"\n{'='*60}")
        print(f"测试策略: {signal_type}")
        print(f"{'='*60}")
        print(f"信号数量: {len(signals)}")
        
        if len(signals) == 0:
            continue
        
        wins = 0
        losses = 0
        total_profit = 0
        
        for signal_idx in signals:
            entry_idx = df.index.get_loc(signal_idx)
            
            # 未来30根K线内的最高/最低价
            future_window = 30
            if entry_idx + future_window >= len(df):
                continue
            
            entry_price = df.iloc[entry_idx]['close']
            future_df = df.iloc[entry_idx+1:entry_idx+future_window+1]
            
            max_profit = (future_df['high'].max() - entry_price) / entry_price * 100
            max_loss = (future_df['low'].min() - entry_price) / entry_price * 100
            
            # 假设止盈1%, 止损0.5%
            tp_pct = 1.0
            sl_pct = -0.5
            
            # 判断盈亏
            if max_profit >= tp_pct:
                wins += 1
                total_profit += tp_pct
            elif max_loss <= sl_pct:
                losses += 1
                total_profit += sl_pct
            else:
                # 30根K线内未触发止盈/止损，按最后价格计算
                final_price = future_df.iloc[-1]['close']
                pnl = (final_price - entry_price) / entry_price * 100
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                total_profit += pnl
        
        total_trades = wins + losses
        if total_trades > 0:
            win_rate = wins / total_trades * 100
            avg_profit = total_profit / total_trades
            
            print(f"\n📊 回测结果:")
            print(f"  总交易次数: {total_trades}")
            print(f"  盈利次数: {wins}")
            print(f"  亏损次数: {losses}")
            print(f"  胜率: {win_rate:.1f}%")
            print(f"  平均盈亏: {avg_profit:+.2f}%")
            print(f"  累计盈亏: {total_profit:+.2f}%")
            
            results[signal_type] = {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'total_profit': total_profit,
            }
    
    # 6. 结论
    print(f"\n{'='*60}")
    print("🎯 结论")
    print(f"{'='*60}")
    
    if results['all_golden_cross']:
        r = results['all_golden_cross']
        print(f"✅ 所有KDJ金叉: 胜率 {r['win_rate']:.1f}% (样本 {r['total_trades']} 笔)")
        
        if r['win_rate'] < 55:
            print("   ⚠️  胜率低于55%，不建议单独使用KDJ金叉")
        elif r['win_rate'] < 60:
            print("   📝 胜率中等，需要配合其他指标过滤")
        else:
            print("   ✨ 胜率较高，但可能存在过拟合，需要样本外验证")
    
    if results['low_golden_cross']:
        r = results['low_golden_cross']
        print(f"✅ 低位金叉(J<20): 胜率 {r['win_rate']:.1f}% (样本 {r['total_trades']} 笔)")

if __name__ == "__main__":
    test_kdj_golden_cross_strategy()
