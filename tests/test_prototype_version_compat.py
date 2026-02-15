"""
测试原型库版本兼容性和迁移逻辑

测试场景：
1. v1.0 原型迁移到 v3.0（计算置信度 + 指纹3D图特征）
2. v2.0 原型迁移到 v3.0（指纹3D图特征）
3. v3.0 原型直接加载（无迁移）
4. 保存后重新加载保持一致性
"""

import sys
import os
import io

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import tempfile
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.template_clusterer import (
    Prototype, 
    PrototypeLibrary, 
    PROTOTYPE_LIBRARY_VERSION,
    PROTOTYPE_MIN_SUPPORTED_VERSION
)


def test_version_constants():
    """测试版本常量定义正确"""
    print("\n=== 测试版本常量 ===")
    
    assert PROTOTYPE_LIBRARY_VERSION == "3.0", f"当前版本应为 3.0, 实际: {PROTOTYPE_LIBRARY_VERSION}"
    assert PROTOTYPE_MIN_SUPPORTED_VERSION == "1.0", f"最低支持版本应为 1.0"
    
    print(f"[OK] 当前版本: {PROTOTYPE_LIBRARY_VERSION}")
    print(f"[OK] 最低支持版本: {PROTOTYPE_MIN_SUPPORTED_VERSION}")


def test_prototype_v1_migration():
    """测试 v1.0 原型迁移到 v3.0"""
    print("\n=== 测试 v1.0 原型迁移 ===")
    
    # 创建模拟 v1.0 原型数据（没有置信度和序列特征）
    v1_data = {
        "prototype_id": 1,
        "direction": "LONG",
        "regime": "强多头",
        "centroid": list(np.random.randn(32)),
        "pre_entry_centroid": list(np.random.randn(32)),
        "holding_centroid": list(np.random.randn(64)),
        "pre_exit_centroid": list(np.random.randn(32)),
        "member_count": 10,
        "win_count": 6,
        "total_profit_pct": 5.0,
        "avg_profit_pct": 0.5,
        "win_rate": 0.6,
        "avg_hold_bars": 30.0,
        # v1.0 没有以下字段:
        # confidence, profit_std, regime_purity
        # weighted_mean, trend_vector, time_segments, volatility
    }
    
    # 从字典还原（应自动触发迁移）
    proto = Prototype.from_dict(v1_data)
    
    # 验证置信度已计算
    assert proto.confidence > 0, "置信度应已计算"
    print(f"[OK] 置信度已计算: {proto.confidence:.2%}")
    
    # 验证 v3.0 特征已迁移
    assert proto.weighted_mean.size == 32, f"weighted_mean 应为 32 维, 实际: {proto.weighted_mean.size}"
    assert proto.trend_vector.size == 32, f"trend_vector 应为 32 维, 实际: {proto.trend_vector.size}"
    assert 'early' in proto.time_segments, "time_segments 应包含 early"
    assert proto.volatility.size == 32, f"volatility 应为 32 维, 实际: {proto.volatility.size}"
    print(f"[OK] v3.0 特征已迁移: weighted_mean={proto.weighted_mean.shape}, trend_vector={proto.trend_vector.shape}")
    
    # 验证 weighted_mean 是 pre_entry_centroid 的加权版本
    # （不是简单复制，应该应用了层权重）
    original = np.array(v1_data["pre_entry_centroid"])
    assert not np.allclose(proto.weighted_mean, original), "weighted_mean 应是加权后的向量"
    print("[OK] weighted_mean 正确应用了层权重")
    
    # 验证版本标记
    assert proto.sequence_features_version == PROTOTYPE_LIBRARY_VERSION
    print(f"[OK] 序列特征版本标记: {proto.sequence_features_version}")


def test_prototype_v2_migration():
    """测试 v2.0 原型迁移到 v3.0"""
    print("\n=== 测试 v2.0 原型迁移 ===")
    
    # 创建模拟 v2.0 原型数据（有置信度但没有序列特征）
    v2_data = {
        "prototype_id": 2,
        "direction": "SHORT",
        "regime": "强空头",
        "centroid": list(np.random.randn(32)),
        "pre_entry_centroid": list(np.random.randn(32)),
        "holding_centroid": list(np.random.randn(64)),
        "pre_exit_centroid": list(np.random.randn(32)),
        "member_count": 15,
        "win_count": 9,
        "total_profit_pct": 7.5,
        "avg_profit_pct": 0.5,
        "win_rate": 0.6,
        "avg_hold_bars": 25.0,
        # v2.0 有置信度字段
        "confidence": 0.75,
        "profit_std": 0.1,
        "regime_purity": 0.9,
        # v2.0 没有以下字段:
        # weighted_mean, trend_vector, time_segments, volatility
    }
    
    proto = Prototype.from_dict(v2_data)
    
    # 验证置信度保持不变（不应被重新计算）
    assert proto.confidence == 0.75, f"置信度应保持 0.75, 实际: {proto.confidence}"
    print(f"[OK] 置信度保持不变: {proto.confidence:.2%}")
    
    # 验证 v3.0 特征已迁移
    assert proto.weighted_mean.size == 32
    assert proto.trend_vector.size == 32
    print(f"[OK] v3.0 特征已迁移")


def test_prototype_v3_no_migration():
    """测试 v3.0 原型无需迁移"""
    print("\n=== 测试 v3.0 原型加载 ===")
    
    # 创建完整 v3.0 原型数据
    v3_data = {
        "prototype_id": 3,
        "direction": "LONG",
        "regime": "震荡偏多",
        "centroid": list(np.random.randn(32)),
        "pre_entry_centroid": list(np.random.randn(32)),
        "holding_centroid": list(np.random.randn(64)),
        "pre_exit_centroid": list(np.random.randn(32)),
        "member_count": 20,
        "win_count": 12,
        "total_profit_pct": 10.0,
        "avg_profit_pct": 0.5,
        "win_rate": 0.6,
        "avg_hold_bars": 35.0,
        "confidence": 0.8,
        "profit_std": 0.08,
        "regime_purity": 0.95,
        "member_fingerprints": ["fp1", "fp2"],
        "verified": True,
        "wf_grade": "合格",
        "wf_match_count": 10,
        "wf_win_rate": 0.7,
        "representative_pre_entry": list(np.random.randn(60, 32).tolist()),
        "member_trade_stats": [(0.5, 30), (0.3, 25)],
        # v3.0 字段
        "weighted_mean": list(np.random.randn(32)),
        "trend_vector": list(np.random.randn(32)),
        "time_segments": {
            "early": list(np.random.randn(32)),
            "mid": list(np.random.randn(32)),
            "late": list(np.random.randn(32)),
        },
        "volatility": list(np.random.randn(32)),
        "sequence_features_version": "3.0",
    }
    
    proto = Prototype.from_dict(v3_data)
    
    # 验证所有字段正确加载
    assert proto.confidence == 0.8
    assert proto.weighted_mean.size == 32
    assert np.allclose(proto.weighted_mean, np.array(v3_data["weighted_mean"]))
    print("[OK] v3.0 原型直接加载，无需迁移")


def test_library_save_load_cycle():
    """测试原型库保存和加载循环"""
    print("\n=== 测试原型库保存/加载循环 ===")
    
    # 创建原型库
    library = PrototypeLibrary()
    library.source_symbol = "BTCUSDT"
    library.source_interval = "1m"
    
    # 添加一些原型
    for i in range(3):
        proto = Prototype(
            prototype_id=i,
            direction="LONG",
            regime="强多头",
            pre_entry_centroid=np.random.randn(32),
            member_count=10,
            win_count=6,
            win_rate=0.6,
        )
        proto.calculate_confidence()
        library.long_prototypes.append(proto)
    
    for i in range(2):
        proto = Prototype(
            prototype_id=i + 100,
            direction="SHORT",
            regime="强空头",
            pre_entry_centroid=np.random.randn(32),
            member_count=8,
            win_count=5,
            win_rate=0.625,
        )
        proto.calculate_confidence()
        library.short_prototypes.append(proto)
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name
    
    try:
        saved_path = library.save(filepath, verbose=True)
        print(f"[OK] 已保存到: {saved_path}")
        
        # 检查文件版本
        version_info = PrototypeLibrary.check_file_version(filepath)
        assert version_info["version"] == PROTOTYPE_LIBRARY_VERSION
        assert not version_info["needs_migration"]
        print(f"[OK] 文件版本检查通过: {version_info['version']}")
        
        # 重新加载
        loaded_library = PrototypeLibrary.load(filepath, verbose=True)
        
        # 验证数据一致性
        assert len(loaded_library.long_prototypes) == 3
        assert len(loaded_library.short_prototypes) == 2
        assert loaded_library.source_symbol == "BTCUSDT"
        print("[OK] 原型库数据一致性验证通过")
        
        # 验证所有原型都有 v3.0 特征
        for proto in loaded_library.get_all_prototypes():
            assert proto.weighted_mean.size == 32, f"原型 {proto.prototype_id} 缺少 weighted_mean"
        print("[OK] 所有原型都有 v3.0 特征")
        
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_library_v2_migration():
    """测试加载 v2.0 原型库自动迁移"""
    print("\n=== 测试 v2.0 原型库迁移 ===")
    
    # 创建模拟 v2.0 原型库文件
    v2_library_data = {
        "version": "2.0",
        "created_at": "2024-01-01T00:00:00",
        "source_template_count": 100,
        "clustering_params": {},
        "source_symbol": "BTCUSDT",
        "source_interval": "1m",
        "long_prototypes": [
            {
                "prototype_id": 1,
                "direction": "LONG",
                "regime": "强多头",
                "pre_entry_centroid": list(np.random.randn(32)),
                "member_count": 10,
                "win_count": 6,
                "win_rate": 0.6,
                "confidence": 0.7,
                "profit_std": 0.1,
                "regime_purity": 0.9,
                # 没有 v3.0 字段
            }
        ],
        "short_prototypes": [],
    }
    
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name
        pickle.dump(v2_library_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    try:
        # 检查版本（应显示需要迁移）
        version_info = PrototypeLibrary.check_file_version(filepath)
        assert version_info["version"] == "2.0"
        assert version_info["needs_migration"]
        assert len(version_info["migration_steps"]) > 0
        print(f"[OK] v2.0 文件检测到需要迁移: {version_info['migration_steps']}")
        
        # 加载并验证自动迁移
        library = PrototypeLibrary.load(filepath, verbose=True)
        
        proto = library.long_prototypes[0]
        assert proto.weighted_mean.size == 32, "迁移后应有 weighted_mean"
        assert proto.sequence_features_version == PROTOTYPE_LIBRARY_VERSION
        print("[OK] v2.0 原型库自动迁移成功")
        
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_get_sequence_features():
    """测试获取序列特征方法"""
    print("\n=== 测试 get_sequence_features() ===")
    
    proto = Prototype(
        prototype_id=1,
        direction="LONG",
        pre_entry_centroid=np.random.randn(32),
        representative_pre_entry=np.random.randn(60, 32),
        member_count=10,
        win_count=6,
        win_rate=0.6,
    )
    
    # 获取序列特征
    features = proto.get_sequence_features()
    
    assert 'raw_sequence' in features
    assert 'weighted_mean' in features
    assert 'trend_vector' in features
    assert 'time_segments' in features
    assert 'volatility' in features
    
    assert features['weighted_mean'].size == 32
    assert features['trend_vector'].size == 32
    print("[OK] get_sequence_features() 返回正确格式")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("原型库版本兼容性测试")
    print("=" * 60)
    
    test_version_constants()
    test_prototype_v1_migration()
    test_prototype_v2_migration()
    test_prototype_v3_no_migration()
    test_library_save_load_cycle()
    test_library_v2_migration()
    test_get_sequence_features()
    
    print("\n" + "=" * 60)
    print("[OK] 所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
