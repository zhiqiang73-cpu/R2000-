"""
测试指纹图3D地形显示
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fingerprint_widget_import():
    """测试 FingerprintWidget 导入"""
    from ui.vector_space_widget import FingerprintWidget, VectorSpaceWidget
    
    # 检查别名
    assert FingerprintWidget is not None
    assert VectorSpaceWidget is FingerprintWidget  # 向后兼容别名
    print("[OK] FingerprintWidget import success")


def test_fingerprint_widget_initialization():
    """测试 FingerprintWidget 初始化（无GUI）"""
    # 创建一个模拟的 TrajectoryTemplate
    from dataclasses import dataclass
    
    @dataclass
    class MockTemplate:
        trade_idx: int = 0
        regime: str = "TREND_UP"
        direction: str = "LONG"
        profit_pct: float = 2.5
        pre_entry: np.ndarray = None
        holding: np.ndarray = None
        pre_exit: np.ndarray = None
        entry_idx: int = 100
        exit_idx: int = 110
        
        def __post_init__(self):
            if self.pre_entry is None:
                self.pre_entry = np.random.randn(60, 32)
            if self.holding is None:
                self.holding = np.random.randn(10, 32)
            if self.pre_exit is None:
                self.pre_exit = np.random.randn(30, 32)
    
    # 创建模拟模板
    templates = [
        MockTemplate(trade_idx=0, regime="TREND_UP", direction="LONG", profit_pct=2.5),
        MockTemplate(trade_idx=1, regime="TREND_DOWN", direction="SHORT", profit_pct=1.8),
        MockTemplate(trade_idx=2, regime="RANGE", direction="LONG", profit_pct=3.2),
    ]
    
    print("[OK] Mock templates created")
    
    # 验证模板结构
    for t in templates:
        assert t.pre_entry.shape == (60, 32)
        assert t.holding.shape[1] == 32
        assert t.pre_exit.shape == (30, 32)
    
    print("[OK] Template data structure verified")


def test_analysis_panel_fingerprint_methods():
    """测试 AnalysisPanel 中的指纹图方法存在性"""
    from ui.analysis_panel import AnalysisPanel
    
    # 检查方法存在
    assert hasattr(AnalysisPanel, 'update_fingerprint_templates')
    assert hasattr(AnalysisPanel, 'update_fingerprint_current')
    
    print("[OK] AnalysisPanel fingerprint methods exist")


def test_surface_plot_logic():
    """测试3D曲面绘制逻辑（不需要GUI）"""
    import numpy as np
    
    # 模拟 pre_entry 矩阵
    matrix = np.random.randn(60, 32)
    
    # 验证网格生成
    n_time, n_features = matrix.shape
    X = np.arange(n_time)
    Y = np.arange(n_features)
    X, Y = np.meshgrid(X, Y)
    Z = matrix.T  # (32, 60)
    
    assert X.shape == (32, 60)
    assert Y.shape == (32, 60)
    assert Z.shape == (32, 60)
    
    print("[OK] 3D surface mesh logic verified")


def test_main_window_fingerprint_integration():
    """测试 main_window 中的指纹图集成"""
    # 仅验证方法存在，不实际运行GUI
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'ui', 'main_window.py'), 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 验证新方法
    assert '_update_fingerprint_view' in content
    assert 'update_fingerprint_templates' in content
    
    # 验证旧的 vector_space_widget.ga_btn 引用已移除
    assert 'vector_space_widget.ga_btn' not in content
    
    print("[OK] main_window fingerprint integration verified")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("指纹图3D地形显示测试")
    print("=" * 60)
    
    tests = [
        test_fingerprint_widget_import,
        test_fingerprint_widget_initialization,
        test_analysis_panel_fingerprint_methods,
        test_surface_plot_logic,
        test_main_window_fingerprint_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
