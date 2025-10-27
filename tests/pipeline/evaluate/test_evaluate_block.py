import pytest
import tempfile
import torch
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.pipeline.step_08_evaluate.evaluate_block import EvaluateBlock
from src.pipeline.io import ArtifactIO
from src.pipeline.schemas import TrainedModelArtifact, SequencesArtifact, ArtifactMetadata


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
        'evaluation': {
            'batch_size': 32,
            'target_signal_rate': 0.05
        }
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_model_artifact(temp_dir):
    """Create sample trained model artifact"""
    # Create dummy model file
    model_path = temp_dir / 'model.pt'
    dummy_model = torch.nn.Linear(10, 3)
    torch.save(dummy_model.state_dict(), model_path)
    
    # Create dummy history file
    history_path = temp_dir / 'history.json'
    history_data = {
        'train_loss': [1.0, 0.8, 0.6],
        'val_loss': [1.1, 0.9, 0.7],
        'train_acc': [0.3, 0.5, 0.7],
        'val_acc': [0.2, 0.4, 0.6]
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f)
    
    return TrainedModelArtifact(
        model_path=model_path,
        history_path=history_path,
        best_val_loss=0.7,
        best_val_acc=0.6,
        total_epochs=3,
        metadata=ArtifactMetadata(schema_name='train')
    )


@pytest.fixture
def sample_sequences_artifact(temp_dir):
    """Create sample sequences artifact"""
    # Create dummy sequence tensors
    val_X = torch.randint(0, 256, (20, 24, 3, 9))  # (samples, seq_len, coins, channels)
    val_y = torch.randint(0, 3, (20,))  # (samples,)
    
    # Save tensors
    val_X_path = temp_dir / 'val_X.pt'
    val_y_path = temp_dir / 'val_y.pt'
    
    torch.save(val_X, val_X_path)
    torch.save(val_y, val_y_path)
    
    return SequencesArtifact(
        train_X_path=temp_dir / 'train_X.pt',  # Dummy path
        train_y_path=temp_dir / 'train_y.pt',  # Dummy path
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


def test_evaluate_block_creates_evaluation_report(config, temp_dir, sample_model_artifact, sample_sequences_artifact):
    """Test that EvaluateBlock creates evaluation report"""
    # Mock the model evaluation
    with patch('src.pipeline.step_08_evaluate.evaluate_block.CryptoTransformerV4') as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock model predictions
        mock_predictions = torch.randn(20, 3)  # (batch_size, num_classes)
        mock_model.return_value = {'horizon_1h': {'logits': mock_predictions}}
        
        # Run evaluation
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = EvaluateBlock(config, artifact_io)
        result = block.run(sample_model_artifact, sample_sequences_artifact)
        
        # Check artifact properties
        assert result.precision > 0
        assert result.recall >= 0
        assert result.f1_score >= 0
        assert result.signal_rate >= 0
        assert result.confusion_matrix is not None
        
        # Check metadata
        assert result.metadata.schema_name == 'evaluate'
        assert result.metadata.timestamp is not None


def test_evaluate_block_handles_model_loading_errors(config, temp_dir, sample_model_artifact, sample_sequences_artifact):
    """Test that EvaluateBlock handles model loading errors gracefully"""
    # Create invalid model file
    invalid_model_path = temp_dir / 'invalid_model.pt'
    invalid_model_path.write_text('invalid model data')
    
    invalid_model_artifact = TrainedModelArtifact(
        model_path=invalid_model_path,
        history_path=sample_model_artifact.history_path,
        best_val_loss=0.7,
        best_val_acc=0.6,
        total_epochs=3,
        metadata=ArtifactMetadata(schema_name='train')
    )
    
    # Run evaluation - should handle error gracefully
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = EvaluateBlock(config, artifact_io)
    
    with pytest.raises(Exception):  # Should raise an error for invalid model
        block.run(invalid_model_artifact, sample_sequences_artifact)


def test_evaluate_block_creates_output_files(config, temp_dir, sample_model_artifact, sample_sequences_artifact):
    """Test that EvaluateBlock creates proper output files"""
    # Mock the model evaluation
    with patch('src.pipeline.step_08_evaluate.evaluate_block.CryptoTransformerV4') as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock model predictions
        mock_predictions = torch.randn(20, 3)
        mock_model.return_value = {'horizon_1h': {'logits': mock_predictions}}
        
        # Run evaluation
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = EvaluateBlock(config, artifact_io)
        result = block.run(sample_model_artifact, sample_sequences_artifact)
        
        # Check that output directory was created
        output_dir = temp_dir / 'step_08_evaluate'
        assert output_dir.exists()
        
        # Check that evaluation report was saved
        assert result.report_path.exists()
        assert result.report_path.parent == output_dir
        
        # Check that confusion matrix plot was created
        assert result.confusion_matrix_path.exists()
        assert result.confusion_matrix_path.parent == output_dir


def test_evaluate_block_calculates_metrics_correctly(config, temp_dir, sample_model_artifact, sample_sequences_artifact):
    """Test that EvaluateBlock calculates metrics correctly"""
    # Mock the model evaluation
    with patch('src.pipeline.step_08_evaluate.evaluate_block.CryptoTransformerV4') as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Create predictable predictions for testing
        mock_predictions = torch.tensor([
            [2.0, 1.0, 0.0],  # Predict class 0
            [1.0, 2.0, 0.0],  # Predict class 1
            [0.0, 1.0, 2.0],  # Predict class 2
            [2.0, 1.0, 0.0],  # Predict class 0
            [1.0, 2.0, 0.0],  # Predict class 1
        ])
        
        # Repeat to match validation set size
        mock_predictions = mock_predictions.repeat(4, 1)  # 20 samples
        
        mock_model.return_value = {'horizon_1h': {'logits': mock_predictions}}
        
        # Run evaluation
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = EvaluateBlock(config, artifact_io)
        result = block.run(sample_model_artifact, sample_sequences_artifact)
        
        # Check that metrics are reasonable
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.f1_score <= 1
        assert 0 <= result.signal_rate <= 1
        
        # Check confusion matrix shape
        assert result.confusion_matrix.shape == (3, 3)  # 3 classes


def test_evaluate_block_metadata_correct(config, temp_dir, sample_model_artifact, sample_sequences_artifact):
    """Test that EvaluateBlock creates correct metadata"""
    # Mock the model evaluation
    with patch('src.pipeline.step_08_evaluate.evaluate_block.CryptoTransformerV4') as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock model predictions
        mock_predictions = torch.randn(20, 3)
        mock_model.return_value = {'horizon_1h': {'logits': mock_predictions}}
        
        # Run evaluation
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = EvaluateBlock(config, artifact_io)
        result = block.run(sample_model_artifact, sample_sequences_artifact)
        
        # Check metadata
        assert result.metadata.schema_name == 'evaluate'
        assert result.metadata.timestamp is not None
        assert 'train_artifact' in result.metadata.upstream_inputs
        assert 'sequences_artifact' in result.metadata.upstream_inputs


def test_evaluate_block_handles_empty_validation_set(config, temp_dir, sample_model_artifact):
    """Test that EvaluateBlock handles empty validation set"""
    # Create sequences artifact with empty validation set
    empty_val_X = torch.empty(0, 24, 3, 9)
    empty_val_y = torch.empty(0, dtype=torch.long)
    
    val_X_path = temp_dir / 'empty_val_X.pt'
    val_y_path = temp_dir / 'empty_val_y.pt'
    
    torch.save(empty_val_X, val_X_path)
    torch.save(empty_val_y, val_y_path)
    
    empty_sequences_artifact = SequencesArtifact(
        train_X_path=temp_dir / 'train_X.pt',
        train_y_path=temp_dir / 'train_y.pt',
        val_X_path=val_X_path,
        val_y_path=val_y_path,
        train_num_samples=100,
        val_num_samples=0,
        input_length=24,
        output_length=1,
        num_coins=3,
        num_channels=9,
        target_coin='XRP',
        metadata=ArtifactMetadata(schema_name='sequences')
    )
    
    # Run evaluation - should handle empty validation set gracefully
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = EvaluateBlock(config, artifact_io)
    
    with pytest.raises(ValueError, match="Validation set is empty"):
        block.run(sample_model_artifact, empty_sequences_artifact)
