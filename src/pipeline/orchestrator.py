"""Pipeline orchestrator for managing step execution and artifact loading"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from src.pipeline.io import ArtifactIO
from src.pipeline.schemas import (
    RawDataArtifact, CleanDataArtifact, SplitDataArtifact, AugmentDataArtifact,
    TokenizeArtifact, SequencesArtifact, TrainedModelArtifact, ArtifactMetadata
)


class PipelineOrchestrator:
    """Orchestrates pipeline execution with artifact management"""
    
    def __init__(self, config: Dict[str, Any], artifact_io: ArtifactIO):
        self.config = config
        self.artifact_io = artifact_io
    
    def load_json_artifact(self, path: str) -> Dict[str, Any]:
        """Load JSON artifact from path"""
        with open(path) as f:
            return json.load(f)
    
    def load_raw_artifact(self, artifact_path: str) -> RawDataArtifact:
        """Load raw data artifact"""
        raw_data = self.load_json_artifact(artifact_path)
        return RawDataArtifact(
            path=Path(raw_data['path']),
            start_date=datetime.fromisoformat(raw_data['start_date']),
            end_date=datetime.fromisoformat(raw_data['end_date']),
            num_timesteps=raw_data['num_timesteps'],
            num_coins=raw_data['num_coins'],
            columns=raw_data['columns'],
            freq=raw_data['freq'],
            metadata=ArtifactMetadata(**raw_data['metadata'])
        )
    
    def load_clean_artifact(self, artifact_path: str) -> CleanDataArtifact:
        """Load clean data artifact"""
        clean_data = self.load_json_artifact(artifact_path)
        return CleanDataArtifact(
            path=Path(clean_data['path']),
            start_date=datetime.fromisoformat(clean_data['start_date']),
            end_date=datetime.fromisoformat(clean_data['end_date']),
            num_timesteps=clean_data['num_timesteps'],
            num_coins=clean_data['num_coins'],
            quality_metrics=clean_data['quality_metrics'],
            metadata=ArtifactMetadata(**clean_data['metadata'])
        )
    
    def load_split_artifact(self, artifact_path: str) -> SplitDataArtifact:
        """Load split data artifact"""
        split_data = self.load_json_artifact(artifact_path)
        return SplitDataArtifact(
            train_path=Path(split_data['train_path']),
            val_path=Path(split_data['val_path']),
            train_samples=split_data['train_samples'],
            val_samples=split_data['val_samples'],
            train_start_date=datetime.fromisoformat(split_data['train_start_date']),
            train_end_date=datetime.fromisoformat(split_data['train_end_date']),
            val_start_date=datetime.fromisoformat(split_data['val_start_date']),
            val_end_date=datetime.fromisoformat(split_data['val_end_date']),
            metadata=ArtifactMetadata(**split_data['metadata'])
        )
    
    def load_augment_artifact(self, artifact_path: str) -> AugmentDataArtifact:
        """Load augment data artifact"""
        augment_data = self.load_json_artifact(artifact_path)
        return AugmentDataArtifact(
            train_path=Path(augment_data['train_path']),
            val_path=Path(augment_data['val_path']),
            train_samples=augment_data['train_samples'],
            val_samples=augment_data['val_samples'],
            num_coins=augment_data['num_coins'],
            indicators_added=augment_data['indicators_added'],
            metadata=ArtifactMetadata(**augment_data['metadata'])
        )
    
    def load_tokenize_artifact(self, artifact_path: str) -> TokenizeArtifact:
        """Load tokenize artifact"""
        tokenize_data = self.load_json_artifact(artifact_path)
        return TokenizeArtifact(
            train_path=Path(tokenize_data['train_path']),
            val_path=Path(tokenize_data['val_path']),
            train_shape=tuple(tokenize_data['train_shape']),
            val_shape=tuple(tokenize_data['val_shape']),
            thresholds_path=Path(tokenize_data['thresholds_path']),
            token_distribution={int(k): v for k, v in tokenize_data['token_distribution'].items()},
            metadata=ArtifactMetadata(**tokenize_data['metadata'])
        )
    
    def load_sequences_artifact(self, artifact_path: str) -> SequencesArtifact:
        """Load sequences artifact"""
        seq_data = self.load_json_artifact(artifact_path)
        return SequencesArtifact(
            train_X_path=Path(seq_data['train_X_path']),
            train_y_path=Path(seq_data['train_y_path']),
            val_X_path=Path(seq_data['val_X_path']),
            val_y_path=Path(seq_data['val_y_path']),
            train_num_samples=seq_data['train_num_samples'],
            val_num_samples=seq_data['val_num_samples'],
            input_length=seq_data['input_length'],
            output_length=seq_data['output_length'],
            num_coins=seq_data['num_coins'],
            num_channels=seq_data.get('num_channels', 2),
            target_coin=seq_data['target_coin'],
            metadata=ArtifactMetadata(**seq_data['metadata'])
        )
    
    def load_trained_model_artifact(self, artifact_path: str) -> TrainedModelArtifact:
        """Load trained model artifact"""
        train_data = self.load_json_artifact(artifact_path)
        return TrainedModelArtifact(
            model_path=Path(train_data['model_path']),
            history_path=Path(train_data['history_path']),
            best_val_loss=train_data['best_val_loss'],
            best_val_acc=train_data['best_val_acc'],
            total_epochs=train_data['total_epochs'],
            metadata=ArtifactMetadata(**train_data['metadata'])
        )
    
    def run_step(self, step_name: str, block_class, *args, **kwargs):
        """Run a pipeline step with error handling"""
        try:
            block = block_class(self.config, self.artifact_io)
            return block.run(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Step {step_name} failed: {e}") from e
