"""
R3000 数据加载模块
负责从 parquet 文件加载数据、随机采样、多时间框架聚合
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, MTF_CONFIG


class DataLoader:
    """
    数据加载器
    - 从 parquet 文件加载 BTCUSDT 1分钟 K 线数据
    - 随机采样连续片段
    - 生成多时间框架数据
    """
    
    def __init__(self, data_file: str = None):
        """
        初始化数据加载器
        
        Args:
            data_file: parquet 文件路径，默认使用配置文件中的路径
        """
        self.data_file = data_file or DATA_CONFIG["DATA_FILE"]
        self.sample_size = DATA_CONFIG["SAMPLE_SIZE"]
        self.warmup_bars = DATA_CONFIG["WARMUP_BARS"]
        
        # 数据缓存
        self._full_data: Optional[pd.DataFrame] = None
        self._sampled_data: Optional[pd.DataFrame] = None
        self._mtf_data: Dict[str, pd.DataFrame] = {}
        
        # 采样信息
        self.sample_start_idx: int = 0
        self.sample_end_idx: int = 0
        self.total_rows: int = 0
        
    def load_full_data(self) -> pd.DataFrame:
        """
        加载完整的 parquet 数据
        
        Returns:
            完整的 K 线数据 DataFrame
        """
        if self._full_data is not None:
            return self._full_data
            
        data_path = Path(self.data_file)
        if not data_path.exists():
            # 尝试相对于脚本目录查找
            script_dir = Path(__file__).parent.parent
            data_path = script_dir / self.data_file
            
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_file}")
            
        print(f"[DataLoader] 正在加载数据: {data_path}")
        self._full_data = pd.read_parquet(data_path)
        self.total_rows = len(self._full_data)
        print(f"[DataLoader] 数据加载完成，共 {self.total_rows:,} 行")
        
        # 确保列名标准化
        self._full_data = self._standardize_columns(self._full_data)
        
        return self._full_data
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名
        """
        # 常见的列名映射
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'open_time': 'timestamp',
            'close_time': 'close_timestamp',
        }
        
        df = df.rename(columns=column_mapping)
        
        # 确保必需的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # 尝试小写匹配
                for c in df.columns:
                    if c.lower() == col:
                        df = df.rename(columns={c: col})
                        break
                        
        # 确保数值类型
        for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
        
    def sample_continuous(self, sample_size: int = None, 
                          random_seed: int = None) -> pd.DataFrame:
        """
        随机采样连续片段
        
        Args:
            sample_size: 采样数量，默认使用配置值
            random_seed: 随机种子
            
        Returns:
            采样后的 K 线数据 DataFrame
        """
        if self._full_data is None:
            self.load_full_data()
            
        sample_size = sample_size or self.sample_size
        
        # 确保采样大小不超过数据量
        max_sample = self.total_rows - self.warmup_bars
        if sample_size > max_sample:
            sample_size = max_sample
            print(f"[DataLoader] 采样大小调整为 {sample_size}")
            
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        elif DATA_CONFIG["RANDOM_SEED"] is not None:
            np.random.seed(DATA_CONFIG["RANDOM_SEED"])
            
        # 随机选择起始位置（确保有足够的预热期）
        max_start = self.total_rows - sample_size
        self.sample_start_idx = np.random.randint(self.warmup_bars, max_start)
        self.sample_end_idx = self.sample_start_idx + sample_size
        
        # 采样
        self._sampled_data = self._full_data.iloc[
            self.sample_start_idx:self.sample_end_idx
        ].copy().reset_index(drop=True)
        
        print(f"[DataLoader] 采样完成: 索引 {self.sample_start_idx:,} - {self.sample_end_idx:,}")
        print(f"[DataLoader] 采样数据: {len(self._sampled_data):,} 行")
        
        # 清除 MTF 缓存
        self._mtf_data = {}
        
        return self._sampled_data
    
    def get_sampled_data(self) -> pd.DataFrame:
        """获取当前采样数据"""
        if self._sampled_data is None:
            return self.sample_continuous()
        return self._sampled_data
    
    def aggregate_timeframe(self, df: pd.DataFrame, 
                            minutes: int) -> pd.DataFrame:
        """
        聚合到更高时间框架
        
        Args:
            df: 1分钟 K 线数据
            minutes: 目标时间框架（分钟）
            
        Returns:
            聚合后的 K 线数据
        """
        if minutes <= 1:
            return df.copy()
            
        # 计算每组的行数
        group_size = minutes
        n_groups = len(df) // group_size
        
        if n_groups == 0:
            return df.copy()
            
        # 截断到完整的组
        df_truncated = df.iloc[:n_groups * group_size].copy()
        
        # 创建分组标签
        df_truncated['group'] = np.arange(len(df_truncated)) // group_size
        
        # 聚合 OHLCV
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        
        # 如果有时间戳列，也保留
        if 'timestamp' in df_truncated.columns:
            agg_dict['timestamp'] = 'first'
            
        aggregated = df_truncated.groupby('group').agg(agg_dict).reset_index(drop=True)
        
        return aggregated
    
    def get_mtf_data(self, timeframe: str = None) -> Dict[str, pd.DataFrame]:
        """
        获取多时间框架数据
        
        Args:
            timeframe: 指定时间框架，None 表示获取所有配置的时间框架
            
        Returns:
            时间框架到 DataFrame 的字典
        """
        if self._sampled_data is None:
            self.sample_continuous()
            
        if timeframe:
            # 获取指定时间框架
            if timeframe not in self._mtf_data:
                minutes = MTF_CONFIG["TIMEFRAMES"].get(timeframe)
                if minutes:
                    self._mtf_data[timeframe] = self.aggregate_timeframe(
                        self._sampled_data, minutes
                    )
            return {timeframe: self._mtf_data.get(timeframe)}
            
        # 获取所有时间框架
        for tf, minutes in MTF_CONFIG["TIMEFRAMES"].items():
            if tf not in self._mtf_data:
                self._mtf_data[tf] = self.aggregate_timeframe(
                    self._sampled_data, minutes
                )
                print(f"[DataLoader] 聚合 {tf} 数据: {len(self._mtf_data[tf]):,} 行")
                
        return self._mtf_data
    
    def get_data_info(self) -> dict:
        """获取数据信息摘要"""
        info = {
            "data_file": self.data_file,
            "total_rows": self.total_rows,
            "sample_size": len(self._sampled_data) if self._sampled_data is not None else 0,
            "sample_start_idx": self.sample_start_idx,
            "sample_end_idx": self.sample_end_idx,
        }
        
        if self._sampled_data is not None and len(self._sampled_data) > 0:
            info["price_range"] = {
                "min": float(self._sampled_data['low'].min()),
                "max": float(self._sampled_data['high'].max()),
            }
            info["volume_total"] = float(self._sampled_data['volume'].sum())
            
            # 时间范围
            if 'timestamp' in self._sampled_data.columns:
                info["time_start"] = str(self._sampled_data['timestamp'].iloc[0])
                info["time_end"] = str(self._sampled_data['timestamp'].iloc[-1])
                
        return info


# 测试代码
if __name__ == "__main__":
    loader = DataLoader()
    
    # 加载并采样
    data = loader.sample_continuous(sample_size=10000, random_seed=42)
    print(f"\n采样数据前5行:\n{data.head()}")
    print(f"\n采样数据列: {data.columns.tolist()}")
    
    # 获取多时间框架数据
    mtf_data = loader.get_mtf_data()
    for tf, df in mtf_data.items():
        print(f"\n{tf} 数据: {len(df)} 行")
        
    # 数据信息
    info = loader.get_data_info()
    print(f"\n数据信息: {info}")
