import pytest
import tempfile
import torch
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.pipeline.step_09_inference.inference_block import InferenceBlock
from src.pipeline.io import ArtifactIO
from src.pipeline.schemas import TrainedModelArtifact, TokenizeArtifact, ArtifactMetadata


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
        'inference': {
            'batch_size': 32,
            'num_samples': 100
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
def sample_tokenize_artifact(temp_dir):
    """Create sample tokenize artifact"""
    # Create dummy tokenized data
    tokenized_data = {
        'timestamp': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
        'BTC_price': [1, 2],
        'BTC_volume': [0, 1],
        'BTC_rsi': [2, 0],
        'BTC_macd': [1, 2],
        'BTC_bb_position': [0, 1],
        'BTC_ema_9': [2, 0],
        'BTC_ema_21': [1, 2],
        'BTC_ema_50': [0, 1],
        'BTC_ema_ratio': [2, 0],
        'ETH_price': [1, 2],
        'ETH_volume': [0, 1],
        'ETH_rsi': [2, 0],
        'ETH_macd': [1, 2],
        'ETH_bb_position': [0, 1],
        'ETH_ema_9': [2, 0],
        'ETH_ema_21': [1, 2],
        'ETH_ema_50': [0, 1],
        'ETH_ema_ratio': [2, 0],
        'XRP_price': [1, 2],
        'XRP_volume': [0, 1],
        'XRP_rsi': [2, 0],
        'XRP_macd': [1, 2],
        'XRP_bb_position': [0, 1],
        'XRP_ema_9': [2, 0],
        'XRP_ema_21': [1, 2],
        'XRP_ema_50': [0, 1],
        'XRP_ema_ratio': [2, 0]
    }
    
    import pandas as pd
    df = pd.DataFrame(tokenized_data)
    
    train_path = temp_dir / 'train_tokens.parquet'
    val_path = temp_dir / 'val_tokens.parquet'
    thresholds_path = temp_dir / 'thresholds.json'
    
    df.to_parquet(train_path)
    df.to_parquet(val_path)
    
    # Create dummy thresholds
    thresholds = {
        'BTC': {
            'price': [-0.01, 0.01],
            'volume': [-0.1, 0.1],
            'rsi': [30, 70],
            'macd': [-0.1, 0.1],
            'bb_position': [0.2, 0.8],
            'ema_9': [-0.05, 0.05],
            'ema_21': [-0.05, 0.05],
            'ema_50': [-0.05, 0.05],
            'ema_ratio': [0.9, 1.1]
        },
        'ETH': {
            'price': [-0.01, 0.01],
            'volume': [-0.1, 0.1],
            'rsi': [30, 70],
            'macd': [-0.1, 0.1],
            'bb_position': [0.2, 0.8],
            'ema_9': [-0.05, 0.05],
            'ema_21': [-0.05, 0.05],
            'ema_50': [-0.05, 0.05],
            'ema_ratio': [0.9, 1.1]
        },
        'XRP': {
            'price': [-0.01, 0.01],
            'volume': [-0.1, 0.1],
            'rsi': [30, 70],
            'macd': [-0.1, 0.1],
            'bb_position': [0.2, 0.8],
            'ema_9': [-0.05, 0.05],
            'ema_21': [-0.05, 0.05],
            'ema_50': [-0.05, 0.05],
            'ema_ratio': [0.9, 1.1]
        }
    }
    
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f)
    
    return TokenizeArtifact(
        train_path=train_path,
        val_path=val_path,
        train_shape=(2, 27),  # 3 coins × 9 channels
        val_shape=(2, 27),
        thresholds_path=thresholds_path,
        token_distribution={0: {'train': 0.33, 'val': 0.30}},
        metadata=ArtifactMetadata(schema_name='tokenize')
    )


def test_inference_block_creates_predictions(config, temp_dir, sample_model_artifact, sample_tokenize_artifact):
    """Test that InferenceBlock creates predictions"""
    # Mock the model inference
    with patch('src.pipeline.step_09_inference.inference_block.CryptoTransformerV4') as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock model predictions
        mock_predictions = torch.randn(100, 3)  # (num_samples, num_classes)
        mock_model.return_value = {'horizon_1h': {'logits': mock_predictions}}
        
        # Run inference
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = InferenceBlock(config, artifact_io)
        result = block.run(sample_model_artifact, sample_tokenize_artifact)
        
        # Check artifact properties
        assert result.predictions_path.exists()
        assert result.num_predictions == 100
        assert result.target_coin == 'XRP'
        
        # Check metadata
        assert result.metadata.schema_name == 'inference'
        assert result.metadata.timestamp is not None


def test_inference_block_handles_model_loading_errors(config, temp_dir, sample_model_artifact, sample_tokenize_artifact):
    """Test that InferenceBlock handles model loading errors gracefully"""
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
    
    # Run inference - should handle error gracefully
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = InferenceBlock(config, artifact_io)
    
    with pytest.raises(Exception):  # Should raise an error for invalid model
        block.run(invalid_model_artifact, sample_tokenize_artifact)


def test_inference_block_creates_output_files(config, temp_dir, sample_model_artifact, sample_tokenize_artifact):
    """Test that InferenceBlock creates proper output files"""
    # Mock the model inference
    with patch('src.pipeline.step_09_inference.inference_block.CryptoTransformerV4') as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock model predictions
        mock_predictions = torch.randn(100, 3)
        mock_model.return_value = {'horizon_1h': {'logits': mock_predictions}}
        
        # Run inference
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = InferenceBlock(config, artifact_io)
        result = block.run(sample_model_artifact, sample_tokenize_artifact)
        
        # Check that output directory was created
        output_dir = temp_dir / 'step_09_inference'
        assert output_dir.exists()
        
        # Check that predictions were saved
        assert result.predictions_path.exists()
        assert result.predictions_path.parent == output_dir
        
        # Check that predictions file contains valid data
        predictions_data = torch.load(result.predictions_path)
        assert isinstance(predictions_data, torch.Tensor)
        assert predictions_data.shape[0] == 100  # num_samples
        assert predictions_data.shape[1] == 3  # num_classes


def test_inference_block_generates_correct_number_of_predictions(config, temp_dir, sample_model_artifact, sample_tokenize_artifact):
    """Test that InferenceBlock generates correct number of predictions"""
    # Mock the model inference
    with patch('src.pipeline.step_09_inference.inference_block.CryptoTransformerV4') as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock model predictions
        mock_predictions = torch.randn(100, 3)
        mock_model.return_value = {'horizon_1h': {'logits': mock_predictions}}
        
        # Run inference
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = InferenceBlock(config, artifact_io)
        result = block.run(sample_model_artifact, sample_tokenize_artifact)
        
        # Check that correct number of predictions were generated
        assert result.num_predictions == 100
        
        # Load predictions and verify shape
        predictions_data = torch.load(result.predictions_path)
        assert predictions_data.shape[0] == 100


def test_inference_block_metadata_correct(config, temp_dir, sample_model_artifact, sample_tokenize_artifact):
    """Test that InferenceBlock creates correct metadata"""
    # Mock the model inference
    with patch('src.pipeline.step_09_inference.inference_block.CryptoTransformerV4') as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock model predictions
        mock_predictions = torch.randn(100, 3)
        mock_model.return_value = {'horizon_1h': {'logits': mock_predictions}}
        
        # Run inference
        artifact_io = ArtifactIO(base_dir=temp_dir)
        block = InferenceBlock(config, artifact_io)
        result = block.run(sample_model_artifact, sample_tokenize_artifact)
        
        # Check metadata
        assert result.metadata.schema_name == 'inference'
        assert result.metadata.timestamp is not None
        assert 'train_artifact' in result.metadata.upstream_inputs
        assert 'tokenize_artifact' in result.metadata.upstream_inputs


def test_inference_block_handles_empty_tokenized_data(config, temp_dir, sample_model_artifact):
    """Test that InferenceBlock handles empty tokenized data"""
    # Create tokenize artifact with empty data
    empty_data = {
        'timestamp': [],
        'BTC_price': [],
        'BTC_volume': [],
        'BTC_rsi': [],
        'BTC_macd': [],
        'BTC_bb_position': [],
        'BTC_ema_9': [],
        'BTC_ema_21': [],
        'BTC_ema_50': [],
        'BTC_ema_ratio': [],
        'ETH_price': [],
        'ETH_volume': [],
        'ETH_rsi': [],
        'ETH_macd': [],
        'ETH_bb_position': [],
        'ETH_ema_9': [],
        'ETH_ema_21': [],
        'ETH_ema_50': [],
        'ETH_ema_ratio': [],
        'XRP_price': [],
        'XRP_volume': [],
        'XRP_rsi': [],
        'XRP_macd': [],
        'XRP_bb_position': [],
        'XRP_ema_9': [],
        'XRP_ema_21': [],
        'XRP_ema_50': [],
        'XRP_ema_ratio': []
    }
    
    import pandas as pd
    df = pd.DataFrame(empty_data)
    
    train_path = temp_dir / 'empty_train_tokens.parquet'
    val_path = temp_dir / 'empty_val_tokens.parquet'
    thresholds_path = temp_dir / 'empty_thresholds.json'
    
    df.to_parquet(train_path)
    df.to_parquet(val_path)
    
    # Create dummy thresholds
    thresholds = {
        'BTC': {'price': [-0.01, 0.01]},
        'ETH': {'price': [-0.01, 0.01]},
        'XRP': {'price': [-0.01, 0.01]}
    }
    
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f)
    
    empty_tokenize_artifact = TokenizeArtifact(
        train_path=train_path,
        val_path=val_path,
        train_shape=(0, 27),
        val_shape=(0, 27),
        thresholds_path=thresholds_path,
        token_distribution={},
        metadata=ArtifactMetadata(schema_name='tokenize')
    )
    
    # Run inference - should handle empty data gracefully
    artifact_io = ArtifactIO(base_dir=temp_dir)
    block = InferenceBlock(config, artifact_io)
    
    with pytest.raises(ValueError, match="No tokenized data available"):
        block.run(sample_model_artifact, empty_tokenize_artifact)
