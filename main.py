"""Refactored main.py - thin CLI with orchestrated pipeline execution"""

import click
import yaml
import sys
from pathlib import Path

from src.pipeline.io import ArtifactIO
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.step_00_reset.reset_block import ResetBlock
from src.pipeline.step_01_download.download_block import DownloadBlock
from src.pipeline.step_02_clean.clean_block import CleanBlock
from src.pipeline.step_03_split.split_block import EarlySplitBlock
from src.pipeline.step_04_augment.augment_block import AugmentBlock
from src.pipeline.step_05_tokenize.tokenize_block import TokenizeBlock
from src.pipeline.step_06_sequences.sequence_block import SequenceBlock
from src.pipeline.step_07_train.train_block import TrainBlock
from src.pipeline.step_08_evaluate.evaluate_block import EvaluateBlock
from src.pipeline.step_09_inference.inference_block import InferenceBlock
from src.utils.config_validator import ConfigValidator
import torch
from src.system.tune import run_tuning


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


def get_orchestrator():
    """Get pipeline orchestrator instance"""
    config = load_config()
    artifact_io = get_artifact_io()
    return PipelineOrchestrator(config, artifact_io)


@click.group()
def cli():
    """Cryptocurrency prediction pipeline"""
    pass


@click.group()
def pipeline():
    """Pipeline commands"""
    pass


# Individual atomic commands
@pipeline.command()
def reset():
    """Reset: Clean artifacts directory"""
    click.echo("\n[STEP 00] RESET: Cleaning artifacts...")
    orchestrator = get_orchestrator()
    orchestrator.run_step("Reset", ResetBlock)
    click.echo("[OK] Reset complete\n")


@pipeline.command()
def download():
    """Download: Fetch raw OHLCV data"""
    click.echo("\n[STEP 01] DOWNLOAD: Fetching cryptocurrency data...")
    orchestrator = get_orchestrator()
    orchestrator.run_step("Download", DownloadBlock)
    click.echo("[OK] Download complete\n")


@pipeline.command()
def clean():
    """Clean: Data cleaning and quality checks"""
    click.echo("\n[STEP 02] CLEAN: Cleaning data...")
    orchestrator = get_orchestrator()
    orchestrator.run_step("Clean", CleanBlock)
    click.echo("[OK] Clean complete\n")


@pipeline.command()
def split():
    """Split: Temporal train/validation split"""
    click.echo("\n[STEP 03] SPLIT: Splitting data temporally...")
    orchestrator = get_orchestrator()
    orchestrator.run_step("Split", EarlySplitBlock)
    click.echo("[OK] Split complete\n")


@pipeline.command()
def augment():
    """Augment: Add technical indicators to split data"""
    click.echo("\n[STEP 04] AUGMENT: Adding technical indicators...")
    orchestrator = get_orchestrator()
    orchestrator.run_step("Augment", AugmentBlock)
    click.echo("[OK] Augment complete\n")


@pipeline.command()
def tokenize():
    """Tokenize: Convert prices to token sequences"""
    click.echo("\n[STEP 05] TOKENIZE: Converting prices to tokens...")
    orchestrator = get_orchestrator()
    orchestrator.run_step("Tokenize", TokenizeBlock)
    click.echo("[OK] Tokenize complete\n")


@pipeline.command()
def train():
    """Train: Train transformer model on sequences"""
    click.echo("\n[STEP 07] TRAIN: Training model...")
    orchestrator = get_orchestrator()
    orchestrator.run_step("Train", TrainBlock)
    click.echo("[OK] Train complete\n")


@pipeline.command()
def evaluate():
    """Evaluate: Validate model quality"""
    click.echo("\n[STEP 08] EVALUATE: Validating model...")
    orchestrator = get_orchestrator()
    orchestrator.run_step("Evaluate", EvaluateBlock)
    click.echo("[OK] Evaluate complete\n")


@pipeline.command()
def inference():
    """Inference: Predict next 8 hours"""
    click.echo("\n[STEP 09] INFERENCE: Predicting next 8 hours...")
    orchestrator = get_orchestrator()
    orchestrator.run_step("Inference", InferenceBlock)
    click.echo("[OK] Inference complete\n")


@pipeline.command()
def sequences():
    """Sequences: Create rolling windows for supervised learning"""
    click.echo("\n[STEP 06] SEQUENCES: Creating rolling windows...")
    orchestrator = get_orchestrator()
    orchestrator.run_step("Sequences", SequenceBlock)
    click.echo("[OK] Sequences complete\n")


# Composite sequence commands
@pipeline.command("run-seq-all")
@click.option('--start-date', default=None, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date (YYYY-MM-DD)')
def run_seq_all(start_date, end_date):
    """Run all pipeline steps"""
    click.echo("\n" + "="*70)
    click.echo("RUNNING COMPLETE PIPELINE")
    click.echo("="*70)
    
    try:
        orchestrator = get_orchestrator()
        
        # Step 00: Reset
        click.echo("\n[STEP 00] RESET...")
        orchestrator.run_step("Reset", ResetBlock)
        
        # Step 01: Download
        click.echo("\n[STEP 01] DOWNLOAD...")
        orchestrator.run_step("Download", DownloadBlock)
        
        # Step 02: Clean
        click.echo("\n[STEP 02] CLEAN...")
        orchestrator.run_step("Clean", CleanBlock)
        
        # Step 03: Split
        click.echo("\n[STEP 03] SPLIT...")
        orchestrator.run_step("Split", EarlySplitBlock)
        
        # Step 04: Augment
        click.echo("\n[STEP 04] AUGMENT...")
        orchestrator.run_step("Augment", AugmentBlock)
        
        # Step 05: Tokenize
        click.echo("\n[STEP 05] TOKENIZE...")
        orchestrator.run_step("Tokenize", TokenizeBlock)
        
        # Step 06: Sequences
        click.echo("\n[STEP 06] SEQUENCES...")
        orchestrator.run_step("Sequences", SequenceBlock)
        
        # Step 07: Train
        click.echo("\n[STEP 07] TRAIN...")
        orchestrator.run_step("Train", TrainBlock)
        
        # Step 08: Evaluate
        click.echo("\n[STEP 08] EVALUATE...")
        orchestrator.run_step("Evaluate", EvaluateBlock)
        
        # Step 09: Inference
        click.echo("\n[STEP 09] INFERENCE...")
        orchestrator.run_step("Inference", InferenceBlock)
        
        click.echo("\n" + "="*70)
        click.echo("PIPELINE COMPLETE - ALL STEPS SUCCESSFUL")
        click.echo("="*70 + "\n")
        
    except Exception as e:
        click.echo(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@pipeline.command("run-seq-from-clean")
def run_seq_from_clean():
    """Run pipeline from clean step (skips reset and download)"""
    click.echo("\n" + "="*70)
    click.echo("RUNNING PIPELINE FROM CLEAN STEP")
    click.echo("(Skipping reset and download - using existing data)")
    click.echo("="*70)
    
    try:
        orchestrator = get_orchestrator()
        
        # Step 02: Clean
        click.echo("\n[STEP 02] CLEAN...")
        raw_artifact = orchestrator.load_raw_artifact('artifacts/step_01_download/raw_data_artifact.json')
        orchestrator.run_step("Clean", CleanBlock, raw_artifact)
        
        # Step 03: Split
        click.echo("\n[STEP 03] SPLIT...")
        clean_artifact = orchestrator.load_clean_artifact('artifacts/step_02_clean/clean_data_artifact.json')
        orchestrator.run_step("Split", EarlySplitBlock, clean_artifact)
        
        # Step 04: Augment
        click.echo("\n[STEP 04] AUGMENT...")
        split_artifact = orchestrator.load_split_artifact('artifacts/step_03_split/split_artifact.json')
        orchestrator.run_step("Augment", AugmentBlock, split_artifact)
        
        # Step 05: Tokenize
        click.echo("\n[STEP 05] TOKENIZE...")
        augment_artifact = orchestrator.load_augment_artifact('artifacts/step_04_augment/augment_artifact.json')
        orchestrator.run_step("Tokenize", TokenizeBlock, augment_artifact)
        
        # Step 06: Sequences
        click.echo("\n[STEP 06] SEQUENCES...")
        tokenize_artifact = orchestrator.load_tokenize_artifact('artifacts/step_05_tokenize/tokenize_artifact.json')
        orchestrator.run_step("Sequences", SequenceBlock, tokenize_artifact)
        
        # Step 07: Train
        click.echo("\n[STEP 07] TRAIN...")
        sequences_artifact = orchestrator.load_sequences_artifact('artifacts/step_06_sequences/sequences_artifact.json')
        orchestrator.run_step("Train", TrainBlock, sequences_artifact)
        
        # Step 08: Evaluate
        click.echo("\n[STEP 08] EVALUATE...")
        train_artifact = orchestrator.load_trained_model_artifact('artifacts/step_07_train/train_artifact.json')
        orchestrator.run_step("Evaluate", EvaluateBlock, train_artifact, sequences_artifact)
        
        # Step 09: Inference
        click.echo("\n[STEP 09] INFERENCE...")
        tokenize_artifact = orchestrator.load_tokenize_artifact('artifacts/step_05_tokenize/tokenize_artifact.json')
        orchestrator.run_step("Inference", InferenceBlock, train_artifact, tokenize_artifact)
        
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
