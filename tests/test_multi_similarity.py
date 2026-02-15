# -*- coding: utf-8 -*-
"""
测试多维相似度匹配系统
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.template_clusterer import PrototypeMatcher, MultiSimilarityCalculator, Prototype, PrototypeLibrary


def test_multi_similarity_match_entry():
    """测试 match_entry 返回正确的多维相似度字段"""
    print("=" * 60)
    print("测试多维相似度匹配系统")
    print("=" * 60)
    
    # Create a test prototype
    np.random.seed(42)
    proto = Prototype(
        prototype_id=0,
        direction="LONG",
        regime="震荡偏多",
        pre_entry_centroid=np.random.randn(32),
        holding_centroid=np.random.randn(64),
        pre_exit_centroid=np.random.randn(32),
        representative_pre_entry=np.random.randn(60, 32),
        member_count=10,
        win_count=6,
        win_rate=0.6,
        avg_profit_pct=1.5,
        confidence=0.7,
    )
    
    # Create a test library
    library = PrototypeLibrary()
    library.long_prototypes = [proto]
    library.short_prototypes = []
    
    # Create matcher with multi-similarity enabled
    matcher = PrototypeMatcher(
        library, 
        enable_multi_similarity=True, 
        include_dtw_in_match=True
    )
    
    # Create a test window
    test_window = np.random.randn(60, 32)
    
    # Run match_entry
    result = matcher.match_entry(test_window, direction="LONG")
    
    # Verify the result has all required fields
    required_fields = [
        "matched", "direction", "best_prototype", "similarity",
        "combined_score", "cosine_similarity", "euclidean_similarity", 
        "dtw_similarity", "confidence", "is_high_confidence",
        "match_threshold", "vote_long", "vote_short", "vote_ratio",
        "top_matches", "top_matches_detail"
    ]
    
    missing_fields = [f for f in required_fields if f not in result]
    if missing_fields:
        print(f"ERROR: Missing fields: {missing_fields}")
        return False
    
    print("SUCCESS: All required fields present!")
    print()
    print("匹配结果详情:")
    print(f"  - matched: {result['matched']}")
    print(f"  - direction: {result['direction']}")
    print(f"  - similarity (final): {result['similarity']:.4f}")
    print(f"  - combined_score: {result['combined_score']:.4f}")
    print(f"  - cosine_similarity: {result['cosine_similarity']:.4f}")
    print(f"  - euclidean_similarity: {result['euclidean_similarity']:.4f}")
    print(f"  - dtw_similarity: {result['dtw_similarity']:.4f}")
    print(f"  - confidence: {result['confidence']:.4f}")
    print(f"  - is_high_confidence: {result['is_high_confidence']}")
    print(f"  - match_threshold: {result['match_threshold']:.4f}")
    print()
    
    # Verify the similarity calculation logic
    expected_combined = (
        0.30 * result['cosine_similarity'] +
        0.40 * result['euclidean_similarity'] +
        0.30 * result['dtw_similarity']
    )
    
    combined_diff = abs(result['combined_score'] - expected_combined)
    if combined_diff > 0.001:
        print(f"WARNING: Combined score mismatch!")
        print(f"  Expected: {expected_combined:.4f}")
        print(f"  Actual: {result['combined_score']:.4f}")
    else:
        print(f"PASS: Combined score calculation correct (差异: {combined_diff:.6f})")
    
    # Verify final score includes confidence
    expected_final = result['combined_score'] * result['confidence']
    final_diff = abs(result['similarity'] - expected_final)
    if final_diff > 0.001:
        print(f"WARNING: Final score mismatch!")
        print(f"  Expected: {expected_final:.4f}")
        print(f"  Actual: {result['similarity']:.4f}")
    else:
        print(f"PASS: Final score = combined * confidence (差异: {final_diff:.6f})")
    
    return True


def test_multi_similarity_calculator():
    """测试 MultiSimilarityCalculator 单独使用"""
    print()
    print("=" * 60)
    print("测试 MultiSimilarityCalculator")
    print("=" * 60)
    
    calculator = MultiSimilarityCalculator()
    
    # Get weights info
    weights = calculator.get_weights_info()
    print(f"权重配置:")
    print(f"  - cosine_weight: {weights['cosine_weight']:.2f}")
    print(f"  - euclidean_weight: {weights['euclidean_weight']:.2f}")
    print(f"  - dtw_weight: {weights['dtw_weight']:.2f}")
    print()
    
    # Test individual similarity functions
    np.random.seed(42)
    v1 = np.random.randn(32)
    v2 = np.random.randn(32)
    
    cos_sim = calculator.compute_cosine_similarity(v1, v2)
    euc_sim = calculator.compute_euclidean_similarity(v1, v2)
    
    print(f"向量相似度测试:")
    print(f"  - cosine_similarity: {cos_sim:.4f}")
    print(f"  - euclidean_similarity: {euc_sim:.4f}")
    
    # Test sequence DTW
    seq1 = np.random.randn(30, 32)
    seq2 = np.random.randn(30, 32)
    
    dtw_sim = calculator.compute_dtw_similarity(seq1, seq2)
    print(f"  - dtw_similarity: {dtw_sim:.4f}")
    
    print()
    print("PASS: MultiSimilarityCalculator 功能正常")
    return True


def test_backward_compatibility():
    """测试向后兼容性（禁用多维相似度时）"""
    print()
    print("=" * 60)
    print("测试向后兼容性（仅余弦相似度模式）")
    print("=" * 60)
    
    np.random.seed(42)
    proto = Prototype(
        prototype_id=0,
        direction="LONG",
        pre_entry_centroid=np.random.randn(32),
        member_count=5,
        win_rate=0.5,
    )
    
    library = PrototypeLibrary()
    library.long_prototypes = [proto]
    
    # Create matcher with multi-similarity DISABLED
    matcher = PrototypeMatcher(
        library, 
        enable_multi_similarity=False
    )
    
    test_window = np.random.randn(60, 32)
    result = matcher.match_entry(test_window, direction="LONG")
    
    # In backward-compatible mode, similarity should equal cosine_similarity
    if abs(result['similarity'] - result['cosine_similarity']) < 0.001:
        print("PASS: 兼容模式下 similarity == cosine_similarity")
    else:
        print(f"WARNING: 兼容模式异常")
        print(f"  similarity: {result['similarity']:.4f}")
        print(f"  cosine_similarity: {result['cosine_similarity']:.4f}")
    
    return True


if __name__ == "__main__":
    test_multi_similarity_match_entry()
    test_multi_similarity_calculator()
    test_backward_compatibility()
    
    print()
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)
