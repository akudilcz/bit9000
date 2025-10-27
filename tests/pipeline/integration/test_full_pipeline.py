import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.io import ArtifactIO


@pytest.fixture
def config():
    """Test configuration for full pipeline"""
    return {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP'],
            'target_coin': 'XRP',
            'data_dir': 'artifacts'
        },
        'split': {
            'train_ratio': 0.8,
            'val_ratio': 0.2
        },
        'model': {
            'vocab_size': 256,
            'num_classes': 3,
            'd_model': 128,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'coin_embedding_dim': 16,
            'max_seq_len': 1024,
            'binary_classification': False,
            'num_channels': 9
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 1,  # Very small for testing
            'patience': 5,
            'min_delta': 0.001
        },
        'evaluation': {
            'batch_size': 32,
            'target_signal_rate': 0.05
        },
        'inference': {
            'batch_size': 32,
            'num_samples': 10  # Small for testing
        }
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing"""
    np.random.seed(42)
    timestamps = pd.date_range('2024-01-01', periods=100, freq='h')
    
    data = {
        'timestamp': timestamps,
    }
    
    for coin in ['BTC', 'ETH', 'XRP']:
        # Price data
        base_price = np.random.uniform(100, 1000)
        returns = np.random.normal(0, 0.02, len(timestamps))
        prices = base_price * np.exp(np.cumsum(returns))
        data[f'{coin}_close'] = prices
        
        # Volume data
        base_volume = np.random.uniform(1000, 10000)
        volume_changes = np.random.normal(0, 0.1, len(timestamps))
        volumes = base_volume * np.exp(np.cumsum(volume_changes))
        data[f'{coin}_volume'] = volumes
        
        # Add other OHLC columns
        data[f'{coin}_open'] = prices * 0.99
        data[f'{coin}_high'] = prices * 1.01
        data[f'{coin}_low'] = prices * 0.98
    
    return pd.DataFrame(data)


def test_full_pipeline_integration(config, temp_dir, sample_raw_data):
    """Test the complete pipeline flow from reset to inference"""
    # Mock the data collector to return our sample data
    with patch('src.pipeline.step_01_download.data_collector.CryptoDataCollector') as mock_collector_class:
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        mock_collector.collect_all_data.return_value = sample_raw_data
        
        # Mock the trainer to avoid actual training
        with patch('src.pipeline.step_07_train.train_block.Trainer') as mock_trainer_class:
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Mock training results
            mock_trainer.train_simple.return_value = {
                'best_val_loss': 0.5,
                'best_val_acc': 0.8,
                'total_epochs': 1,
                'model_path': temp_dir / 'model.pt',
                'history_path': temp_dir / 'history.json'
            }
            
            # Mock model evaluation
            with patch('src.pipeline.step_08_evaluate.evaluate_block.CryptoTransformerV4') as mock_model_class:
                mock_model = Mock()
                mock_model_class.return_value = mock_model
                mock_model.return_value = {'horizon_1h': {'logits': torch.randn(20, 3)}}
                
                # Mock inference model
                with patch('src.pipeline.step_09_inference.inference_block.CryptoTransformerV4') as mock_inference_model_class:
                    mock_inference_model = Mock()
                    mock_inference_model_class.return_value = mock_inference_model
                    mock_inference_model.return_value = {'horizon_1h': {'logits': torch.randn(10, 3)}}
                    
                    # Create orchestrator
                    artifact_io = ArtifactIO(base_dir=temp_dir)
                    orchestrator = PipelineOrchestrator(config, artifact_io)
                    
                    # Run full pipeline
                    orchestrator.run_all_pipeline()
                    
                    # Verify that all steps completed successfully
                    artifacts_dir = temp_dir / 'artifacts'
                    
                    # Check that all step directories exist
                    for step in range(10):  # steps 00-09
                        step_dir = artifacts_dir / f'step_{step:02d}_reset' if step == 0 else artifacts_dir / f'step_{step:02d}_{["reset", "download", "clean", "split", "augment", "tokenize", "sequences", "train", "evaluate", "inference"][step]}'
                        assert step_dir.exists(), f"Step {step} directory should exist"
                    
                    # Check specific artifacts
                    assert (artifacts_dir / 'step_01_download' / 'raw_data.parquet').exists()
                    assert (artifacts_dir / 'step_02_clean' / 'clean_data.parquet').exists()
                    assert (artifacts_dir / 'step_03_split' / 'train.parquet').exists()
                    assert (artifacts_dir / 'step_03_split' / 'val.parquet').exists()
                    assert (artifacts_dir / 'step_04_augment' / 'train_augmented.parquet').exists()
                    assert (artifacts_dir / 'step_04_augment' / 'val_augmented.parquet').exists()
                    assert (artifacts_dir / 'step_05_tokenize' / 'train_tokens.parquet').exists()
                    assert (artifacts_dir / 'step_05_tokenize' / 'val_tokens.parquet').exists()
                    assert (artifacts_dir / 'step_06_sequences' / 'train_X.pt').exists()
                    assert (artifacts_dir / 'step_06_sequences' / 'train_y.pt').exists()
                    assert (artifacts_dir / 'step_06_sequences' / 'val_X.pt').exists()
                    assert (artifacts_dir / 'step_06_sequences' / 'val_y.pt').exists()
                    assert (artifacts_dir / 'step_07_train' / 'model.pt').exists()
                    assert (artifacts_dir / 'step_08_evaluate' / 'eval_report.json').exists()
                    assert (artifacts_dir / 'step_09_inference' / 'predictions.pt').exists()


def test_pipeline_from_clean_integration(config, temp_dir, sample_raw_data):
    """Test the pipeline flow from clean step onwards"""
    # First, create some initial artifacts (simulate steps 0-2)
    artifacts_dir = temp_dir / 'artifacts'
    artifacts_dir.mkdir()
    
    # Create dummy raw data
    raw_data_path = artifacts_dir / 'step_01_download' / 'raw_data.parquet'
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    sample_raw_data.to_parquet(raw_data_path)
    
    # Create dummy clean data
    clean_data_path = artifacts_dir / 'step_02_clean' / 'clean_data.parquet'
    clean_data_path.parent.mkdir(parents=True, exist_ok=True)
    sample_raw_data.to_parquet(clean_data_path)
    
    # Mock the trainer
    with patch('src.pipeline.step_07_train.train_block.Trainer') as mock_trainer_class:
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock training results
        mock_trainer.train_simple.return_value = {
            'best_val_loss': 0.5,
            'best_val_acc': 0.8,
            'total_epochs': 1,
            'model_path': temp_dir / 'model.pt',
            'history_path': temp_dir / 'history.json'
        }
        
        # Mock model evaluation
        with patch('src.pipeline.step_08_evaluate.evaluate_block.CryptoTransformerV4') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_model.return_value = {'horizon_1h': {'logits': torch.randn(20, 3)}}
            
            # Mock inference model
            with patch('src.pipeline.step_09_inference.inference_block.CryptoTransformerV4') as mock_inference_model_class:
                mock_inference_model = Mock()
                mock_inference_model_class.return_value = mock_inference_model
                mock_inference_model.return_value = {'horizon_1h': {'logits': torch.randn(10, 3)}}
                
                # Create orchestrator
                artifact_io = ArtifactIO(base_dir=temp_dir)
                orchestrator = PipelineOrchestrator(config, artifact_io)
                
                # Run pipeline from clean
                orchestrator.run_from_clean_pipeline()
                
                # Verify that steps 3-9 completed successfully
                for step in range(3, 10):
                    step_name = ["", "", "", "split", "augment", "tokenize", "sequences", "train", "evaluate", "inference"][step]
                    step_dir = artifacts_dir / f'step_{step:02d}_{step_name}'
                    assert step_dir.exists(), f"Step {step} directory should exist"


def test_pipeline_error_handling(config, temp_dir):
    """Test that pipeline handles errors gracefully"""
    # Create orchestrator
    artifact_io = ArtifactIO(base_dir=temp_dir)
    orchestrator = PipelineOrchestrator(config, artifact_io)
    
    # Mock download to raise an error
    with patch('src.pipeline.step_01_download.data_collector.CryptoDataCollector') as mock_collector_class:
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        mock_collector.collect_all_data.side_effect = Exception("Download failed")
        
        # Run pipeline - should handle error gracefully
        with pytest.raises(Exception, match="Download failed"):
            orchestrator.run_all_pipeline()


def test_pipeline_artifact_dependencies(config, temp_dir, sample_raw_data):
    """Test that pipeline maintains proper artifact dependencies"""
    # Mock the data collector
    with patch('src.pipeline.step_01_download.data_collector.CryptoDataCollector') as mock_collector_class:
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        mock_collector.collect_all_data.return_value = sample_raw_data
        
        # Mock the trainer
        with patch('src.pipeline.step_07_train.train_block.Trainer') as mock_trainer_class:
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            mock_trainer.train_simple.return_value = {
                'best_val_loss': 0.5,
                'best_val_acc': 0.8,
                'total_epochs': 1,
                'model_path': temp_dir / 'model.pt',
                'history_path': temp_dir / 'history.json'
            }
            
            # Mock model evaluation and inference
            with patch('src.pipeline.step_08_evaluate.evaluate_block.CryptoTransformerV4'), \
                 patch('src.pipeline.step_09_inference.inference_block.CryptoTransformerV4'):
                
                # Create orchestrator
                artifact_io = ArtifactIO(base_dir=temp_dir)
                orchestrator = PipelineOrchestrator(config, artifact_io)
                
                # Run full pipeline
                orchestrator.run_all_pipeline()
                
                # Check that artifact metadata contains proper upstream dependencies
                artifacts_dir = temp_dir / 'artifacts'
                
                # Check split artifact metadata
                split_artifact_path = artifacts_dir / 'step_03_split' / 'split_artifact.json'
                if split_artifact_path.exists():
                    import json
                    with open(split_artifact_path, 'r') as f:
                        split_artifact = json.load(f)
                    assert 'clean_data' in split_artifact['metadata']['upstream_inputs']
                
                # Check augment artifact metadata
                augment_artifact_path = artifacts_dir / 'step_04_augment' / 'augment_artifact.json'
                if augment_artifact_path.exists():
                    with open(augment_artifact_path, 'r') as f:
                        augment_artifact = json.load(f)
                    assert 'split_artifact' in augment_artifact['metadata']['upstream_inputs']
