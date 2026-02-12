"""临时脚本：完成paper_trader.py的限价单功能修改"""

# 读取文件
with open(r'core\paper_trader.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. 在reset方法后添加辅助方法（找到 print("[PaperTrader] 账户已重置") 这一行）
helper_methods = '''    
    def _market_close(self, price: float, bar_idx: int, reason: CloseReason) -> PaperOrder:
        """市价紧急平仓"""
        order = self.current_position
        actual_price = price
        order.close(exit_price=actual_price, exit_time=datetime.now(), exit_bar_idx=bar_idx, reason=reason, leverage=self.leverage)
        notional = order.quantity * actual_price
        fee = notional * self.taker_fee_rate
        pnl = order.realized_pnl - fee
        self.balance += pnl
        self._update_stats(order)
        if order.template_fingerprint:
            self._record_template_performance(order)
        self.order_history.append(order)
        self.current_position = None
        print(f"[PaperTrader] 市价平仓: {reason.value} @ {actual_price:.2f} | 盈亏: {order.profit_pct:+.2f}%")
        if self.on_trade_closed:
            self.on_trade_closed(order)
        return order
    
    def _check_limit_order_fill(self, price: float, high: float, low: float) -> bool:
        """检查限价单是否成交"""
        if self.current_position is None or not self.current_position.pending_limit_order:
            return False
        order = self.current_position
        limit_price = order.limit_order_price
        if order.side == OrderSide.LONG:
            return high >= limit_price
        else:
            return low <= limit_price
    
    def _execute_limit_order_fill(self, bar_idx: int) -> PaperOrder:
        """执行限价单成交"""
        order = self.current_position
        actual_price = order.limit_order_price
        reason = order.close_reason or CloseReason.MANUAL
        order.close(exit_price=actual_price, exit_time=datetime.now(), exit_bar_idx=bar_idx, reason=reason, leverage=self.leverage)
        notional = order.quantity * actual_price
        fee = notional * self.maker_fee_rate
        pnl = order.realized_pnl - fee
        self.balance += pnl
        self._update_stats(order)
        if order.template_fingerprint:
            self._record_template_performance(order)
        self.order_history.append(order)
        self.current_position = None
        print(f"[PaperTrader] 限价单成交: {reason.value} @ {actual_price:.2f} | 盈亏: {order.profit_pct:+.2f}% (Maker,无滑点)")
        if self.on_trade_closed:
            self.on_trade_closed(order)
        return order
    
    def _cancel_and_relist_limit_order(self, current_price: float, bar_idx: int):
        """重新挂单"""
        order = self.current_position
        old_price = order.limit_order_price
        print(f"[PaperTrader] 限价单超时 @ {old_price:.2f}，重挂...")
        if order.side == OrderSide.LONG:
            new_limit_price = current_price * (1 + self.limit_order_offset)
        else:
            new_limit_price = current_price * (1 - self.limit_order_offset)
        order.limit_order_price = new_limit_price
        order.limit_order_start_bar = bar_idx
        print(f"[PaperTrader] 重新挂限价单: @ {new_limit_price:.2f}")

'''

# 找到插入位置
insert_idx = None
for i, line in enumerate(lines):
    if '[PaperTrader] 账户已重置' in line:
        insert_idx = i + 1
        break

if insert_idx:
    lines.insert(insert_idx, helper_methods)
    print(f'OK: Insert helper methods after line {insert_idx}')

# 2. 修改update_price方法 - 在止盈止损检查前添加限价单检查
# 找到 def update_price 开始
update_price_start = None
for i, line in enumerate(lines):
    if 'def update_price(self, price: float' in line:
        update_price_start = i
        break

if update_price_start:
    # 找到 "# 更新未实现盈亏" 这一行
    for i in range(update_price_start, min(update_price_start + 50, len(lines))):
        if '# 更新未实现盈亏' in lines[i] or 'order.update_pnl(price' in lines[i]:
            # 在这之前插入限价单检查
            limit_check_code = '''        
        # 【优先检查限价单】
        if order.pending_limit_order:
            filled = self._check_limit_order_fill(price, high, low)
            if filled:
                self._execute_limit_order_fill(bar_idx or self.current_bar_idx)
                return order.close_reason
            else:
                if bar_idx is not None and (bar_idx - order.limit_order_start_bar) >= self.limit_order_max_wait:
                    self._cancel_and_relist_limit_order(price, bar_idx)
            order.update_pnl(price, self.leverage)
            if self.on_order_update:
                self.on_order_update(order)
            return None
        
'''
            lines[i] = limit_check_code + lines[i]
            print(f'OK: Add limit order check at line {i}')
            break

# 3. 修改close_position调用，添加use_limit_order=False
for i, line in enumerate(lines):
    if 'self.close_position(' in line and 'CloseReason.TAKE_PROFIT' in line:
        lines[i] = line.replace('CloseReason.TAKE_PROFIT)', 'CloseReason.TAKE_PROFIT, use_limit_order=False)')
    if 'self.close_position(' in line and 'CloseReason.STOP_LOSS' in line:
        lines[i] = line.replace('CloseReason.STOP_LOSS)', 'CloseReason.STOP_LOSS, use_limit_order=False)')
    if 'self.close_position(' in line and 'CloseReason.DERAIL' in line:
        lines[i] = line.replace('CloseReason.DERAIL)', 'CloseReason.DERAIL, use_limit_order=False)')

print('OK: Modified close_position calls')

# 写回文件
with open(r'core\paper_trader.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('\nDONE: All modifications complete!')
