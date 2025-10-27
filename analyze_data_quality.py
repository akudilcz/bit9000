"""
Data Quality Analysis for Pipeline Stages 0-4
Analyzes artifacts from: reset, download, clean, split, augment
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

class DataQualityAnalyzer:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.results = {}
    
    def load_artifact(self, path):
        """Load a JSON artifact"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def analyze_download(self):
        """Analyze raw data from download stage"""
        print("\n" + "="*70)
        print("STAGE 1: DOWNLOAD - Raw Data Quality")
        print("="*70)
        
        artifact_path = self.artifacts_dir / "step_01_download" / "raw_data_artifact.json"
        if not artifact_path.exists():
            print("❌ Download artifact not found")
            return
        
        artifact = self.load_artifact(artifact_path)
        
        # Summary stats
        print(f"\n📊 Raw Data Summary:")
        print(f"  Path: {artifact['path']}")
        print(f"  Timesteps: {artifact['num_timesteps']:,}")
        print(f"  Coins: {artifact['num_coins']}")
        print(f"  Start Date: {artifact['start_date']}")
        print(f"  End Date: {artifact['end_date']}")
        print(f"  Frequency: {artifact['freq']}")
        
        # Load and inspect the dataframe
        try:
            df = pd.read_parquet(artifact['path'])
            print(f"\n📈 DataFrame Shape: {df.shape}")
            print(f"  Columns: {len(df.columns)}")
            print(f"\n📋 Column Names:")
            for col in sorted(df.columns):
                print(f"    - {col}")
            
            # Data quality metrics
            print(f"\n🔍 Data Quality Metrics:")
            print(f"  Null values: {df.isna().sum().sum():,}")
            print(f"  Null rate: {df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
            print(f"  Duplicated rows: {df.index.duplicated().sum()}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            
            # Per-coin stats
            print(f"\n💰 Per-Coin Statistics:")
            coins = list(set([col.replace('_close', '').replace('_open', '').replace('_high', '')
                             .replace('_low', '').replace('_volume', '') for col in df.columns]))
            for coin in sorted(coins):
                close_col = f"{coin}_close"
                if close_col in df.columns:
                    close_data = df[close_col].dropna()
                    print(f"  {coin}:")
                    print(f"    Non-null values: {len(close_data):,} / {len(df):,}")
                    print(f"    Price range: ${close_data.min():.2f} - ${close_data.max():.2f}")
            
            self.results['download'] = {
                'shape': df.shape,
                'null_rate': df.isna().sum().sum() / (df.shape[0] * df.shape[1]),
                'df': df
            }
            
        except Exception as e:
            print(f"❌ Error loading dataframe: {e}")
    
    def analyze_clean(self):
        """Analyze cleaned data"""
        print("\n" + "="*70)
        print("STAGE 2: CLEAN - Data Cleaning Quality")
        print("="*70)
        
        artifact_path = self.artifacts_dir / "step_02_clean" / "clean_data_artifact.json"
        if not artifact_path.exists():
            print("❌ Clean artifact not found")
            return
        
        artifact = self.load_artifact(artifact_path)
        
        print(f"\n📊 Cleaned Data Summary:")
        print(f"  Path: {artifact['path']}")
        print(f"  Timesteps: {artifact['num_timesteps']:,}")
        print(f"  Coins: {artifact['num_coins']}")
        print(f"  Start Date: {artifact['start_date']}")
        print(f"  End Date: {artifact['end_date']}")
        
        # Quality metrics
        print(f"\n📈 Quality Metrics:")
        if 'quality_metrics' in artifact:
            for key, value in artifact['quality_metrics'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # Load and compare
        try:
            df_clean = pd.read_parquet(artifact['path'])
            print(f"\n📋 DataFrame Shape: {df_clean.shape}")
            print(f"  Null values: {df_clean.isna().sum().sum()}")
            print(f"  Null rate: {df_clean.isna().sum().sum() / (df_clean.shape[0] * df_clean.shape[1]) * 100:.4f}%")
            
            if 'download' in self.results:
                df_raw = self.results['download']['df']
                print(f"\n🔄 Changes from Raw:")
                print(f"  Rows removed: {len(df_raw) - len(df_clean):,}")
                print(f"  Rows retained: {len(df_clean):,}")
                print(f"  Retention rate: {len(df_clean) / len(df_raw) * 100:.2f}%")
            
            self.results['clean'] = {'shape': df_clean.shape, 'df': df_clean}
            
        except Exception as e:
            print(f"❌ Error loading dataframe: {e}")
    
    def analyze_split(self):
        """Analyze train/val split"""
        print("\n" + "="*70)
        print("STAGE 3: SPLIT - Train/Val Temporal Split")
        print("="*70)
        
        artifact_path = self.artifacts_dir / "step_03_split" / "split_artifact.json"
        if not artifact_path.exists():
            print("❌ Split artifact not found")
            return
        
        artifact = self.load_artifact(artifact_path)
        
        print(f"\n📊 Split Summary:")
        print(f"  Train samples: {artifact['train_samples']:,}")
        print(f"  Val samples: {artifact['val_samples']:,}")
        print(f"  Total samples: {artifact['train_samples'] + artifact['val_samples']:,}")
        print(f"  Train/Val ratio: {artifact['train_samples'] / artifact['val_samples']:.2f}")
        
        print(f"\n📅 Temporal Distribution:")
        print(f"  Train period: {artifact['train_start_date']} to {artifact['train_end_date']}")
        print(f"  Val period: {artifact['val_start_date']} to {artifact['val_end_date']}")
        
        # Load and verify
        try:
            df_train = pd.read_parquet(artifact['train_path'])
            df_val = pd.read_parquet(artifact['val_path'])
            
            print(f"\n📈 DataFrame Shapes:")
            print(f"  Train: {df_train.shape}")
            print(f"  Val: {df_val.shape}")
            
            print(f"\n🔍 Data Quality:")
            print(f"  Train null rate: {df_train.isna().sum().sum() / (df_train.shape[0] * df_train.shape[1]) * 100:.4f}%")
            print(f"  Val null rate: {df_val.isna().sum().sum() / (df_val.shape[0] * df_val.shape[1]) * 100:.4f}%")
            
            self.results['split'] = {
                'train_shape': df_train.shape,
                'val_shape': df_val.shape,
                'df_train': df_train,
                'df_val': df_val
            }
            
        except Exception as e:
            print(f"❌ Error loading dataframes: {e}")
    
    def analyze_augment(self):
        """Analyze augmented data with indicators"""
        print("\n" + "="*70)
        print("STAGE 4: AUGMENT - Technical Indicators")
        print("="*70)
        
        artifact_path = self.artifacts_dir / "step_04_augment" / "augment_artifact.json"
        if not artifact_path.exists():
            print("❌ Augment artifact not found")
            return
        
        artifact = self.load_artifact(artifact_path)
        
        print(f"\n📊 Augmented Data Summary:")
        print(f"  Train samples: {artifact['train_samples']:,}")
        print(f"  Val samples: {artifact['val_samples']:,}")
        print(f"  Coins: {artifact['num_coins']}")
        
        print(f"\n🔧 Technical Indicators Added:")
        if 'indicators_added' in artifact:
            for indicator in artifact['indicators_added']:
                print(f"  - {indicator}")
        
        # Load and inspect
        try:
            df_train = pd.read_parquet(artifact['train_path'])
            df_val = pd.read_parquet(artifact['val_path'])
            
            print(f"\n📈 DataFrame Shapes:")
            print(f"  Train: {df_train.shape}")
            print(f"  Val: {df_val.shape}")
            print(f"  Columns added: {df_train.shape[1] - self.results.get('split', {}).get('train_shape', (0,1))[1] if 'split' in self.results else 'N/A'}")
            
            print(f"\n📋 New Columns (Sample):")
            if 'split' in self.results:
                df_split_train = self.results['split']['df_train']
                new_cols = [col for col in df_train.columns if col not in df_split_train.columns]
                for col in sorted(new_cols)[:15]:
                    print(f"  - {col}")
                if len(new_cols) > 15:
                    print(f"  ... and {len(new_cols) - 15} more")
            
            print(f"\n🔍 Data Quality:")
            print(f"  Train null rate: {df_train.isna().sum().sum() / (df_train.shape[0] * df_train.shape[1]) * 100:.4f}%")
            print(f"  Val null rate: {df_val.isna().sum().sum() / (df_val.shape[0] * df_val.shape[1]) * 100:.4f}%")
            
            self.results['augment'] = {
                'train_shape': df_train.shape,
                'val_shape': df_val.shape,
                'df_train': df_train,
                'df_val': df_val
            }
            
        except Exception as e:
            print(f"❌ Error loading dataframes: {e}")
    
    def summary_report(self):
        """Print summary report"""
        print("\n" + "="*70)
        print("DATA QUALITY SUMMARY")
        print("="*70)
        
        stages_completed = [k for k in ['download', 'clean', 'split', 'augment'] if k in self.results]
        print(f"\n✅ Stages Completed: {', '.join(stages_completed) if stages_completed else 'None'}")
        
        if 'augment' in self.results:
            print(f"\n📊 Final Data Summary:")
            print(f"  Train shape: {self.results['augment']['train_shape']}")
            print(f"  Val shape: {self.results['augment']['val_shape']}")
            
            # Estimate memory usage
            df_train = self.results['augment']['df_train']
            memory_mb = df_train.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"  Estimated memory (train): {memory_mb:.2f} MB")
    
    def run(self):
        """Run all analyses"""
        print("\n" + "="*70)
        print("DATA QUALITY ANALYSIS - Stages 0-4")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        self.analyze_download()
        self.analyze_clean()
        self.analyze_split()
        self.analyze_augment()
        self.summary_report()
        
        print("\n" + "="*70)
        print(f"Analysis Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")

if __name__ == "__main__":
    analyzer = DataQualityAnalyzer()
    analyzer.run()
