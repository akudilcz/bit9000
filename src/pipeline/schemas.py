"""Pydantic schemas for pipeline artifacts"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import numpy as np


class ArtifactMetadata(BaseModel):
    """Metadata for all pipeline artifacts"""
    schema_name: str
    schema_version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
    config_hash: Optional[str] = None
    upstream_inputs: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class ResetArtifact(BaseModel):
    """Reset block artifact - tracks cleanup operation"""
    reset_timestamp: datetime
    artifacts_cleaned: bool
    cleaned_directories: List[str]
    metadata: ArtifactMetadata
    
    class Config:
        arbitrary_types_allowed = True


class RawDataArtifact(BaseModel):
    """Raw OHLCV data from download block"""
    path: Path
    start_date: datetime
    end_date: datetime
    num_timesteps: int
    num_coins: int
    columns: List[str]
    freq: str = "H"
    metadata: ArtifactMetadata
    
    @field_validator('columns')
    @classmethod
    def validate_columns(cls, v):
        required_suffixes = ['_open', '_high', '_low', '_close', '_volume']
        for suffix in required_suffixes:
            if not any(col.endswith(suffix) for col in v):
                raise ValueError(f"Missing required column suffix: {suffix}")
        return v


class CleanDataArtifact(BaseModel):
    """Cleaned data from clean block"""
    path: Path
    start_date: datetime
    end_date: datetime
    num_timesteps: int
    num_coins: int
    quality_metrics: Dict[str, float]  # nan_rate, dup_count, gap_count, etc.
    metadata: ArtifactMetadata


class SplitDataArtifact(BaseModel):
    """Early train/val split artifact from step_03_split"""
    train_path: Path
    val_path: Path
    train_samples: int
    val_samples: int
    train_start_date: datetime
    train_end_date: datetime
    val_start_date: datetime
    val_end_date: datetime
    metadata: ArtifactMetadata


class TokenizeArtifact(BaseModel):
    """Tokenized data from tokenize block (Step 4)"""
    train_path: Path
    val_path: Path
    train_shape: Tuple[int, int]  # (timesteps, num_coins)
    val_shape: Tuple[int, int]  # (timesteps, num_coins)
    thresholds_path: Path  # Per-coin quantile thresholds
    token_distribution: Dict[int, Dict[str, float]]  # {0: {train: 0.33, val: 0.28}, ...}
    metadata: ArtifactMetadata


class SequencesArtifact(BaseModel):
    """Sequences (X, y) from sequence block (Step 5)"""
    train_X_path: Path
    train_y_path: Path
    val_X_path: Path
    val_y_path: Path
    train_num_samples: int
    val_num_samples: int
    input_length: int  # 48 hours
    output_length: int  # 8 hours
    num_coins: int
    target_coin: str  # "XRP"
    metadata: ArtifactMetadata


class TrainedModelArtifact(BaseModel):
    """Trained model from train block (Step 6)"""
    model_path: Path
    history_path: Path
    best_val_loss: float
    best_val_acc: float
    total_epochs: int
    metadata: ArtifactMetadata


class EvalReportArtifact(BaseModel):
    """Evaluation report from evaluate block (Step 7)"""
    report_path: Path
    per_hour_accuracy: List[float]  # Accuracy for hours 1-8
    sequence_accuracy: float  # All 8 correct
    per_class_metrics: Dict[str, Dict[str, float]]  # hour -> {down: {p, r, f1}, ...}
    baseline_comparison: Dict[str, Dict[str, float]]  # {persistence: {h1: 0.38, ...}, ...}
    metadata: ArtifactMetadata


class InferenceArtifact(BaseModel):
    """Inference results from inference block (Step 8)"""
    prediction_path: Path
    timestamp: datetime
    predictions: List[Dict[str, Any]]  # List of 8 hourly predictions
    metadata: ArtifactMetadata


class TuneArtifact(BaseModel):
    """Hyperparameter tuning results (optional Step 6A)"""
    best_params: Dict[str, Any]
    best_val_loss: float
    best_trial: int
    num_trials: int
    results_path: Path
    trials_path: Path
    plots_dir: Path


class PretrainArtifact(BaseModel):
    """Pretraining results on synthetic data (optional Step 6A)"""
    model_path: Path
    history_path: Path
    stats_path: Path
    num_synthetic_samples: int
    best_val_loss: float
    best_val_acc: float
    total_epochs: int


