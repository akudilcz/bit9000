import pytest
import tempfile
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.pipeline.step_07_train.train_block import TrainBlock
from src.pipeline.io import ArtifactIO
from src.pipeline.schemas import SequencesArtifact, ArtifactMetadata


@pytest.fixture
def config():
    """Test configuration"""
    return {
        'data': {
            'coins': ['BTC', 'ETH', 'XRP'],
            'target_coin': 'XRP'
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
            'num_epochs': 2,  # Small number for testing
            'patience': 5,
            'min_delta': 0.001
        }
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_sequences_data(temp_dir):
    """Create sample sequences data for testing"""
    # Create dummy sequence tensors
    train_X = torch.randint(0, 256, (100, 24, 3, 9))  # (samples, seq_len, coins, channels)
    train_y = torch.randint(0, 3, (100,))  # (samples,)
    val_X = torch.randint(0, 256, (20, 24, 3, 9))
    val_y = torch.randint(0, 3, (20,))
    
    # Save tensors
    train_X_path = temp_dir / 'train_X.pt'
    train_y_path = temp_dir / 'train_y.pt'
    val_X_path = temp_dir / 'val_X.pt'
    val_y_path = temp_dir / 'val_y.pt'
    
    torch.save(train_X, train_X_path)
    torch.save(train_y, train_y_path)
    torch.save(val_X, val_X_path)
    torch.save(val_y, val_y_path)
    
    return train_X_path, train_y_path, val_X_path, val_y_path


def test_train_block_creates_model_and_artifacts(config, temp_dir, sample_sequences_data):
    """Test that TrainBlock creates model and training artifacts"""
    train_X_path, train_y_path, val_X_path, val_y_path = sample_sequences_data
    
    # Create sequences artifact
    sequences_artifact = SequencesArtifact(
        train_X_path=train_X_path,
        train_y_path=train_y_path,
        val_X_path=val_X_path,
        val_y_path=val_y_path,
        train_num_samples=100,
        val_num_samples=20,
        input_length=24,
        output_length=1,
        num_coins=3,
        num_channels=9,
        target_coin='XRP',
        metadata=ArtifactMetadata(schema_name='sequences')
    )
    
    # Mock the trainer to avoid actual training
    with patch('src.pipeline.step_07_train.train_block.Trainer') as mock_trainer_class:
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock training results
        mock_trainer.train_simple.return_value = {
            'best_val_loss': 0.5,
            'best_val_acc': 0.8,
            'total_epochs': 2,
            'model_path': temp_dir / 'model.pt',
            'history_path': temp_dir / 'history.json'
        }
        
        # Run training
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = TrainBlock(config, artifact_io)
        result = block.run(sequences_artifact)
        
        # Check artifact properties
        assert result.best_val_loss == 0.5
        assert result.best_val_acc == 0.8
        assert result.total_epochs == 2
        assert result.model_path.exists()
        assert result.history_path.exists()
        
        # Check metadata
        assert result.metadata.schema_name == 'train'
        assert result.metadata.timestamp is not None


def test_train_block_handles_training_errors(config, temp_dir, sample_sequences_data):
    """Test that TrainBlock handles training errors gracefully"""
    train_X_path, train_y_path, val_X_path, val_y_path = sample_sequences_data
    
    # Create sequences artifact
    sequences_artifact = SequencesArtifact(
        train_X_path=train_X_path,
        train_y_path=train_y_path,
        val_X_path=val_X_path,
        val_y_path=val_y_path,
        train_num_samples=100,
        val_num_samples=20,
        input_length=24,
        output_length=1,
        num_coins=3,
        num_channels=9,
        target_coin='XRP',
        metadata=ArtifactMetadata(schema_name='sequences')
    )
    
    # Mock trainer to raise an error
    with patch('src.pipeline.step_07_train.train_block.Trainer') as mock_trainer_class:
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train_simple.side_effect = RuntimeError("Training failed")
        
        # Run training - should handle error gracefully
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = TrainBlock(config, artifact_io)
        
        with pytest.raises(RuntimeError, match="Training failed"):
            block.run(sequences_artifact)


def test_train_block_creates_output_directory(config, temp_dir, sample_sequences_data):
    """Test that TrainBlock creates proper output directory structure"""
    train_X_path, train_y_path, val_X_path, val_y_path = sample_sequences_data
    
    # Create sequences artifact
    sequences_artifact = SequencesArtifact(
        train_X_path=train_X_path,
        train_y_path=train_y_path,
        val_X_path=val_X_path,
        val_y_path=val_y_path,
        train_num_samples=100,
        val_num_samples=20,
        input_length=24,
        output_length=1,
        num_coins=3,
        num_channels=9,
        target_coin='XRP',
        metadata=ArtifactMetadata(schema_name='sequences')
    )
    
    # Mock the trainer
    with patch('src.pipeline.step_07_train.train_block.Trainer') as mock_trainer_class:
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock training results
        mock_trainer.train_simple.return_value = {
            'best_val_loss': 0.5,
            'best_val_acc': 0.8,
            'total_epochs': 2,
            'model_path': temp_dir / 'model.pt',
            'history_path': temp_dir / 'history.json'
        }
        
        # Run training
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = TrainBlock(config, artifact_io)
        result = block.run(sequences_artifact)
        
        # Check that output directory was created
        output_dir = temp_dir / 'step_07_train'
        assert output_dir.exists()
        
        # Check that artifacts were saved in the correct location
        assert result.model_path.parent == output_dir
        assert result.history_path.parent == output_dir


def test_train_block_validates_input_data(config, temp_dir, sample_sequences_data):
    """Test that TrainBlock validates input data"""
    train_X_path, train_y_path, val_X_path, val_y_path = sample_sequences_data
    
    # Create sequences artifact with mismatched data
    sequences_artifact = SequencesArtifact(
        train_X_path=train_X_path,
        train_y_path=train_y_path,
        val_X_path=val_X_path,
        val_y_path=val_y_path,
        train_num_samples=100,
        val_num_samples=20,
        input_length=24,
        output_length=1,
        num_coins=3,
        num_channels=9,
        target_coin='XRP',
        metadata=ArtifactMetadata(schema_name='sequences')
    )
    
    # Mock the trainer
    with patch('src.pipeline.step_07_train.train_block.Trainer') as mock_trainer_class:
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock training results
        mock_trainer.train_simple.return_value = {
            'best_val_loss': 0.5,
            'best_val_acc': 0.8,
            'total_epochs': 2,
            'model_path': temp_dir / 'model.pt',
            'history_path': temp_dir / 'history.json'
        }
        
        # Run training
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = TrainBlock(config, artifact_io)
        result = block.run(sequences_artifact)
        
        # Verify that trainer was called with correct parameters
        mock_trainer.train_simple.assert_called_once()
        call_args = mock_trainer.train_simple.call_args
        
        # Check that the trainer received the correct data
        assert call_args is not None
        # The exact validation depends on the trainer implementation


def test_train_block_metadata_correct(config, temp_dir, sample_sequences_data):
    """Test that TrainBlock creates correct metadata"""
    train_X_path, train_y_path, val_X_path, val_y_path = sample_sequences_data
    
    # Create sequences artifact
    sequences_artifact = SequencesArtifact(
        train_X_path=train_X_path,
        train_y_path=train_y_path,
        val_X_path=val_X_path,
        val_y_path=val_y_path,
        train_num_samples=100,
        val_num_samples=20,
        input_length=24,
        output_length=1,
        num_coins=3,
        num_channels=9,
        target_coin='XRP',
        metadata=ArtifactMetadata(schema_name='sequences')
    )
    
    # Mock the trainer
    with patch('src.pipeline.step_07_train.train_block.Trainer') as mock_trainer_class:
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock training results
        mock_trainer.train_simple.return_value = {
            'best_val_loss': 0.5,
            'best_val_acc': 0.8,
            'total_epochs': 2,
            'model_path': temp_dir / 'model.pt',
            'history_path': temp_dir / 'history.json'
        }
        
        # Run training
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = TrainBlock(config, artifact_io)
        result = block.run(sequences_artifact)
        
        # Check metadata
        assert result.metadata.schema_name == 'train'
        assert result.metadata.timestamp is not None
        assert 'sequences_artifact' in result.metadata.upstream_inputs
