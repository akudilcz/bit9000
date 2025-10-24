#!/usr/bin/env python3
"""Main entry point for cryptocurrency prediction pipeline"""

import click
import yaml
import sys
import json
from pathlib import Path
from datetime import datetime

from src.pipeline.io import ArtifactIO
from src.pipeline.step_00_reset.reset_block import ResetBlock
from src.pipeline.step_01_download.download_block import DownloadBlock
from src.pipeline.step_02_clean.clean_block import CleanBlock
from src.pipeline.step_03_split.split_block import EarlySplitBlock
from src.pipeline.step_04_tokenize.tokenize_block import TokenizeBlock, TokenizeArtifact
from src.pipeline.step_05_sequences.sequence_block import SequenceBlock, SequencesArtifact
from src.pipeline.step_06_train.train_block import TrainBlock, TrainedModelArtifact
from src.pipeline.step_07_evaluate.evaluate_block import EvaluateBlock
from src.pipeline.step_08_inference.inference_block import InferenceBlock
from src.pipeline.schemas import (
    RawDataArtifact, CleanDataArtifact, SplitDataArtifact,
    ArtifactMetadata
)
from src.utils.config_validator import ConfigValidator


def load_config():
    """Load and validate configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate config has all required values
    ConfigValidator.validate(config, strict=False)
    
    # Log critical config values
    ConfigValidator.log_critical_values(config)
    
    return config


def get_artifact_io():
    """Get artifact IO instance"""
    return ArtifactIO(base_dir='artifacts')


def load_json_artifact(path: str):
    """Load JSON artifact from path"""
    with open(path) as f:
        return json.load(f)


@click.group()
def cli():
    """Cryptocurrency prediction pipeline"""
    pass


@click.group()
def pipeline():
    """Pipeline commands"""
    pass


@pipeline.command()
def reset():
    """Reset: Clean artifacts directory"""
    click.echo("\n[STEP 00] RESET: Cleaning artifacts...")
    config = load_config()
    artifact_io = get_artifact_io()
    block = ResetBlock(config, artifact_io)
    block.run()
    click.echo("[OK] Reset complete\n")


@pipeline.command()
@click.option('--start-date', default=None, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date (YYYY-MM-DD)')
def download(start_date, end_date):
    """Download: Fetch raw OHLCV data"""
    click.echo("\n[STEP 01] DOWNLOAD: Fetching cryptocurrency data...")
    config = load_config()
    artifact_io = get_artifact_io()
    block = DownloadBlock(config, artifact_io)
    block.run(start_date=start_date, end_date=end_date)
    click.echo("[OK] Download complete\n")


@pipeline.command()
def clean():
    """Clean: Data cleaning and quality checks"""
    click.echo("\n[STEP 02] CLEAN: Cleaning data...")
    config = load_config()
    artifact_io = get_artifact_io()
    raw_data = load_json_artifact('artifacts/step_01_download/raw_data_artifact.json')
    raw_artifact = RawDataArtifact(
        path=Path(raw_data['path']),
        start_date=datetime.fromisoformat(raw_data['start_date']),
        end_date=datetime.fromisoformat(raw_data['end_date']),
        num_timesteps=raw_data['num_timesteps'],
        num_coins=raw_data['num_coins'],
        columns=raw_data['columns'],
        freq=raw_data['freq'],
        metadata=ArtifactMetadata(**raw_data['metadata'])
    )
    block = CleanBlock(config, artifact_io)
    block.run(raw_artifact)
    click.echo("[OK] Clean complete\n")


@pipeline.command()
def split():
    """Split: Temporal train/validation split"""
    click.echo("\n[STEP 03] SPLIT: Splitting data temporally...")
    config = load_config()
    artifact_io = get_artifact_io()
    clean_data = load_json_artifact('artifacts/step_02_clean/clean_data_artifact.json')
    clean_artifact = CleanDataArtifact(
        path=Path(clean_data['path']),
        start_date=datetime.fromisoformat(clean_data['start_date']),
        end_date=datetime.fromisoformat(clean_data['end_date']),
        num_timesteps=clean_data['num_timesteps'],
        num_coins=clean_data['num_coins'],
        quality_metrics=clean_data['quality_metrics'],
        metadata=ArtifactMetadata(**clean_data['metadata'])
    )
    block = EarlySplitBlock(config, artifact_io)
    block.run(clean_artifact)
    click.echo("[OK] Split complete\n")


@pipeline.command()
def tokenize():
    """Tokenize: Convert prices to token sequences"""
    click.echo("\n[STEP 04] TOKENIZE: Converting prices to tokens...")
    config = load_config()
    artifact_io = get_artifact_io()
    split_data = load_json_artifact('artifacts/step_03_split/split_artifact.json')
    split_artifact = SplitDataArtifact(
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
    block = TokenizeBlock(config, artifact_io)
    block.run(split_artifact)
    click.echo("[OK] Tokenize complete\n")


@pipeline.command()
def sequences():
    """Sequences: Create rolling windows for supervised learning"""
    click.echo("\n[STEP 05] SEQUENCES: Creating rolling windows...")
    config = load_config()
    artifact_io = get_artifact_io()
    
    # Load tokenize artifact
    tokenize_data = load_json_artifact('artifacts/step_04_tokenize/tokenize_artifact.json')
    tokenize_artifact = TokenizeArtifact(
        train_path=Path(tokenize_data['train_path']),
        val_path=Path(tokenize_data['val_path']),
        train_shape=tuple(tokenize_data['train_shape']),
        val_shape=tuple(tokenize_data['val_shape']),
        thresholds_path=Path(tokenize_data['thresholds_path']),
        token_distribution={int(k): v for k, v in tokenize_data['token_distribution'].items()},
        metadata=ArtifactMetadata(**tokenize_data['metadata'])
    )
    
    block = SequenceBlock(config, artifact_io)
    block.run(tokenize_artifact)
    click.echo("[OK] Sequences complete\n")


@pipeline.command()
def train():
    """Train: Train transformer model on sequences"""
    click.echo("\n[STEP 06] TRAIN: Training model...")
    config = load_config()
    artifact_io = get_artifact_io()
    
    # Load sequences artifact
    seq_data = load_json_artifact('artifacts/step_05_sequences/sequences_artifact.json')
    sequences_artifact = SequencesArtifact(
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
    
    block = TrainBlock(config, artifact_io)
    block.run(sequences_artifact)
    click.echo("[OK] Train complete\n")


@pipeline.command()
def evaluate():
    """Evaluate: Validate model quality"""
    click.echo("\n[STEP 07] EVALUATE: Validating model...")
    config = load_config()
    artifact_io = get_artifact_io()
    
    # Load train artifact
    train_data = load_json_artifact('artifacts/step_06_train/train_artifact.json')
    train_artifact = TrainedModelArtifact(
        model_path=Path(train_data['model_path']),
        history_path=Path(train_data['history_path']),
        best_val_loss=train_data['best_val_loss'],
        best_val_acc=train_data['best_val_acc'],
        total_epochs=train_data['total_epochs'],
        metadata=ArtifactMetadata(**train_data['metadata'])
    )
    
    # Load sequences artifact
    seq_data = load_json_artifact('artifacts/step_05_sequences/sequences_artifact.json')
    sequences_artifact = SequencesArtifact(
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
    
    block = EvaluateBlock(config, artifact_io)
    block.run(train_artifact, sequences_artifact)
    click.echo("[OK] Evaluate complete\n")


@pipeline.command()
def inference():
    """Inference: Predict next 8 hours"""
    click.echo("\n[STEP 08] INFERENCE: Predicting next 8 hours...")
    config = load_config()
    artifact_io = get_artifact_io()
    
    # Load train artifact
    train_data = load_json_artifact('artifacts/step_06_train/train_artifact.json')
    train_artifact = TrainedModelArtifact(
        model_path=Path(train_data['model_path']),
        history_path=Path(train_data['history_path']),
        best_val_loss=train_data['best_val_loss'],
        best_val_acc=train_data['best_val_acc'],
        total_epochs=train_data['total_epochs'],
        metadata=ArtifactMetadata(**train_data['metadata'])
    )
    
    # Load tokenize artifact (for thresholds)
    tokenize_data = load_json_artifact('artifacts/step_04_tokenize/tokenize_artifact.json')
    tokenize_artifact = TokenizeArtifact(
        train_path=Path(tokenize_data['train_path']),
        val_path=Path(tokenize_data['val_path']),
        train_shape=tuple(tokenize_data['train_shape']),
        val_shape=tuple(tokenize_data['val_shape']),
        thresholds_path=Path(tokenize_data['thresholds_path']),
        token_distribution={int(k): v for k, v in tokenize_data['token_distribution'].items()},
        metadata=ArtifactMetadata(**tokenize_data['metadata'])
    )
    
    block = InferenceBlock(config, artifact_io)
    block.run(train_artifact, tokenize_artifact)
    click.echo("[OK] Inference complete\n")


@pipeline.command()
@click.option('--start-date', default=None, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date (YYYY-MM-DD)')
def run_all(start_date, end_date):
    """Run all pipeline steps"""
    click.echo("\n" + "="*70)
    click.echo("RUNNING COMPLETE PIPELINE")
    click.echo("="*70)
    
    try:
        # Step 00: Reset
        click.echo("\n[STEP 00] RESET...")
        config = load_config()
        artifact_io = get_artifact_io()
        block = ResetBlock(config, artifact_io)
        block.run()
        
        # Step 01: Download
        click.echo("\n[STEP 01] DOWNLOAD...")
        block = DownloadBlock(config, artifact_io)
        block.run(start_date=start_date, end_date=end_date)
        
        # Step 02: Clean
        click.echo("\n[STEP 02] CLEAN...")
        raw_data = load_json_artifact('artifacts/step_01_download/raw_data_artifact.json')
        raw_artifact = RawDataArtifact(
            path=Path(raw_data['path']),
            start_date=datetime.fromisoformat(raw_data['start_date']),
            end_date=datetime.fromisoformat(raw_data['end_date']),
            num_timesteps=raw_data['num_timesteps'],
            num_coins=raw_data['num_coins'],
            columns=raw_data['columns'],
            freq=raw_data['freq'],
            metadata=ArtifactMetadata(**raw_data['metadata'])
        )
        block = CleanBlock(config, artifact_io)
        block.run(raw_artifact)
        
        # Step 03: Split
        click.echo("\n[STEP 03] SPLIT...")
        clean_data = load_json_artifact('artifacts/step_02_clean/clean_data_artifact.json')
        clean_artifact = CleanDataArtifact(
            path=Path(clean_data['path']),
            start_date=datetime.fromisoformat(clean_data['start_date']),
            end_date=datetime.fromisoformat(clean_data['end_date']),
            num_timesteps=clean_data['num_timesteps'],
            num_coins=clean_data['num_coins'],
            quality_metrics=clean_data['quality_metrics'],
            metadata=ArtifactMetadata(**clean_data['metadata'])
        )
        block = EarlySplitBlock(config, artifact_io)
        block.run(clean_artifact)
        
        # Step 04: Tokenize
        click.echo("\n[STEP 04] TOKENIZE...")
        split_data = load_json_artifact('artifacts/step_03_split/split_artifact.json')
        split_artifact = SplitDataArtifact(
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
        block = TokenizeBlock(config, artifact_io)
        block.run(split_artifact)
        
        # Step 05: Sequences
        click.echo("\n[STEP 05] SEQUENCES...")
        tokenize_data = load_json_artifact('artifacts/step_04_tokenize/tokenize_artifact.json')
        tokenize_artifact = TokenizeArtifact(
            train_path=Path(tokenize_data['train_path']),
            val_path=Path(tokenize_data['val_path']),
            train_shape=tuple(tokenize_data['train_shape']),
            val_shape=tuple(tokenize_data['val_shape']),
            thresholds_path=Path(tokenize_data['thresholds_path']),
            token_distribution={int(k): v for k, v in tokenize_data['token_distribution'].items()},
            metadata=ArtifactMetadata(**tokenize_data['metadata'])
        )
        block = SequenceBlock(config, artifact_io)
        block.run(tokenize_artifact)
        
        # Step 06: Train
        click.echo("\n[STEP 06] TRAIN...")
        seq_data = load_json_artifact('artifacts/step_05_sequences/sequences_artifact.json')
        sequences_artifact = SequencesArtifact(
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
        block = TrainBlock(config, artifact_io)
        block.run(sequences_artifact)
        
        click.echo("\n" + "="*70)
        click.echo("PIPELINE COMPLETE - ALL STEPS SUCCESSFUL")
        click.echo("="*70 + "\n")
        
    except Exception as e:
        click.echo(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@pipeline.command()
def run_from_clean():
    """Run pipeline from clean step (skips reset and download)"""
    click.echo("\n" + "="*70)
    click.echo("RUNNING PIPELINE FROM CLEAN STEP")
    click.echo("(Skipping reset and download - using existing data)")
    click.echo("="*70)
    
    try:
        config = load_config()
        artifact_io = get_artifact_io()
        
        # Step 02: Clean
        click.echo("\n[STEP 02] CLEAN...")
        raw_data = load_json_artifact('artifacts/step_01_download/raw_data_artifact.json')
        raw_artifact = RawDataArtifact(
            path=Path(raw_data['path']),
            start_date=datetime.fromisoformat(raw_data['start_date']),
            end_date=datetime.fromisoformat(raw_data['end_date']),
            num_timesteps=raw_data['num_timesteps'],
            num_coins=raw_data['num_coins'],
            columns=raw_data['columns'],
            freq=raw_data['freq'],
            metadata=ArtifactMetadata(**raw_data['metadata'])
        )
        block = CleanBlock(config, artifact_io)
        block.run(raw_artifact)
        
        # Step 03: Split
        click.echo("\n[STEP 03] SPLIT...")
        clean_data = load_json_artifact('artifacts/step_02_clean/clean_data_artifact.json')
        clean_artifact = CleanDataArtifact(
            path=Path(clean_data['path']),
            start_date=datetime.fromisoformat(clean_data['start_date']),
            end_date=datetime.fromisoformat(clean_data['end_date']),
            num_timesteps=clean_data['num_timesteps'],
            num_coins=clean_data['num_coins'],
            quality_metrics=clean_data['quality_metrics'],
            metadata=ArtifactMetadata(**clean_data['metadata'])
        )
        block = EarlySplitBlock(config, artifact_io)
        block.run(clean_artifact)
        
        # Step 04: Tokenize
        click.echo("\n[STEP 04] TOKENIZE...")
        split_data = load_json_artifact('artifacts/step_03_split/split_artifact.json')
        split_artifact = SplitDataArtifact(
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
        block = TokenizeBlock(config, artifact_io)
        block.run(split_artifact)
        
        # Step 05: Sequences
        click.echo("\n[STEP 05] SEQUENCES...")
        tokenize_data = load_json_artifact('artifacts/step_04_tokenize/tokenize_artifact.json')
        tokenize_artifact = TokenizeArtifact(
            train_path=Path(tokenize_data['train_path']),
            val_path=Path(tokenize_data['val_path']),
            train_shape=tuple(tokenize_data['train_shape']),
            val_shape=tuple(tokenize_data['val_shape']),
            thresholds_path=Path(tokenize_data['thresholds_path']),
            token_distribution={int(k): v for k, v in tokenize_data['token_distribution'].items()},
            metadata=ArtifactMetadata(**tokenize_data['metadata'])
        )
        block = SequenceBlock(config, artifact_io)
        block.run(tokenize_artifact)
        
        # Step 06: Train
        click.echo("\n[STEP 06] TRAIN...")
        seq_data = load_json_artifact('artifacts/step_05_sequences/sequences_artifact.json')
        sequences_artifact = SequencesArtifact(
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
        block = TrainBlock(config, artifact_io)
        block.run(sequences_artifact)
        
        click.echo("\n" + "="*70)
        click.echo("PIPELINE COMPLETE - ALL STEPS SUCCESSFUL")
        click.echo("="*70 + "\n")
        
    except Exception as e:
        click.echo(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Register subcommands
cli.add_command(pipeline)


if __name__ == '__main__':
    cli()
