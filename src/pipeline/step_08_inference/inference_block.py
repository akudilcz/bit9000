"""Step 8: Inference - Load model and predict next 8 hours

Philosophy: Deploy for real predictions
- Fetch last 24 hours of data (price + volume)
- Tokenize using saved thresholds (2 channels)
- Run autoregressive inference → next 8 tokens
- Return predictions with probabilities
"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

from src.pipeline.base import PipelineBlock
from src.pipeline.schemas import ArtifactMetadata
from src.model.token_predictor import SimpleTokenPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceArtifact:
    """Artifact for inference results"""
    def __init__(self, predictions_path: Path, timestamp: datetime,
                 predictions: List[Dict], metadata: ArtifactMetadata):
        self.predictions_path = predictions_path
        self.timestamp = timestamp
        self.predictions = predictions
        self.metadata = metadata
    
    def model_dump(self, mode='json'):
        return {
            'predictions_path': str(self.predictions_path),
            'timestamp': self.timestamp.isoformat(),
            'predictions': self.predictions,
            'metadata': self.metadata.model_dump(mode=mode)
        }


class InferenceBlock(PipelineBlock):
    """Run inference on latest data"""
    
    def run(self, train_artifact, tokenize_artifact):
        """
        Run inference to predict next 8 hours
        
        Args:
            train_artifact: TrainedModelArtifact from step_06_train
            tokenize_artifact: TokenizeArtifact from step_04_tokenize (for thresholds)
            
        Returns:
            InferenceArtifact
        """
        logger.info("="*70)
        logger.info("STEP 8: INFERENCE - Predicting next 8 hours")
        logger.info("="*70)
        
        device = self.config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        target_coin = self.config['data']['target_coin']
        input_length = self.config['sequences']['input_length']
        output_length = self.config['sequences']['output_length']
        
        logger.info(f"\n  Target coin: {target_coin}")
        logger.info(f"  Input length: {input_length} hours")
        logger.info(f"  Output length: {output_length} hours")
        
        # Load model
        logger.info("\n[1/5] Loading trained model...")
        model = SimpleTokenPredictor(self.config)
        checkpoint = torch.load(train_artifact.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        logger.info(f"  Model loaded from: {train_artifact.model_path}")
        
        # Load thresholds
        logger.info("\n[2/5] Loading tokenization thresholds...")
        with open(tokenize_artifact.thresholds_path, 'r') as f:
            thresholds = json.load(f)
        logger.info(f"  Loaded thresholds for {len(thresholds)} coins")
        
        # Fetch latest data (last input_length hours + 1 for log returns)
        logger.info("\n[3/5] Fetching latest data...")
        latest_data = self._fetch_latest_data(input_length + 1)
        logger.info(f"  Fetched data: {latest_data.shape}")
        logger.info(f"  Date range: {latest_data.index[0]} to {latest_data.index[-1]}")
        
        # Tokenize (2 channels: price + volume)
        logger.info("\n[4/5] Tokenizing latest data (price + volume)...")
        tokens = self._tokenize_latest(latest_data, thresholds)
        logger.info(f"  Tokens: {tokens.shape} (timesteps × coins × channels)")
        
        # Verify we have exactly input_length timesteps
        if len(tokens) != input_length:
            raise ValueError(f"Expected {input_length} tokens, got {len(tokens)}")
        
        # Run autoregressive inference
        logger.info("\n[5/5] Running autoregressive inference...")
        predictions, probabilities = self._predict(model, tokens, device, output_length)
        
        logger.info(f"  Predictions: {predictions}")
        logger.info(f"  Probabilities shape: {probabilities.shape}")
        
        # Format results
        timestamp = datetime.now()
        results = self._format_results(predictions, probabilities, timestamp, target_coin, output_length)
        
        # Log predictions
        logger.info("\n  Predicted sequence:")
        for pred in results['predictions']:
            logger.info(f"    Hour {pred['hour']}: {pred['prediction']} (confidence: {pred['confidence']:.2f})")
        
        # Save results
        block_dir = self.artifact_io.get_block_dir("step_08_inference", clean=True)
        predictions_path = block_dir / f"predictions_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(predictions_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n  Saved predictions: {predictions_path}")
        
        # Create artifact
        artifact = InferenceArtifact(
            predictions_path=predictions_path,
            timestamp=timestamp,
            predictions=results['predictions'],
            metadata=self.create_metadata(
                upstream_inputs={
                    "model": str(train_artifact.model_path),
                    "thresholds": str(tokenize_artifact.thresholds_path)
                }
            )
        )
        
        # Write artifact manifest
        self.artifact_io.write_json(
            artifact.model_dump(mode='json'),
            block_name="step_08_inference",
            artifact_name="inference_artifact"
        )
        
        logger.info("\n" + "="*70)
        logger.info("INFERENCE COMPLETE")
        logger.info(f"  Timestamp: {timestamp}")
        logger.info(f"  Predictions: {output_length} hours")
        logger.info("="*70 + "\n")
        
        return artifact
    
    def _fetch_latest_data(self, num_hours: int) -> pd.DataFrame:
        """
        Fetch latest data from artifacts
        
        For real deployment, this would fetch from API.
        For now, we use the last N hours from clean data.
        
        Args:
            num_hours: Number of hours to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        # Load clean data (would be API call in production)
        clean_path = Path("artifacts/step_02_clean/clean_data.parquet")
        if not clean_path.exists():
            raise FileNotFoundError(f"Clean data not found: {clean_path}")
        
        df = pd.read_parquet(clean_path)
        
        # Take last N hours
        latest = df.tail(num_hours).copy()
        
        return latest
    
    def _tokenize_latest(self, df: pd.DataFrame, thresholds: Dict[str, Dict[str, tuple]]) -> np.ndarray:
        """
        Tokenize latest data using fitted thresholds (2 channels)
        
        Args:
            df: DataFrame with COIN_close and COIN_volume columns
            thresholds: Fitted thresholds {coin: {price: (tau_low, tau_high), volume: (tau_low, tau_high)}}
            
        Returns:
            Array of tokens, shape (timesteps, num_coins, 2)
            Channel 0: price tokens
            Channel 1: volume tokens
        """
        coins = self.config['data']['coins']
        num_timesteps = len(df) - 1  # After computing returns
        num_coins = len(coins)
        
        # Initialize output array: (timesteps, num_coins, 2)
        tokens_array = np.zeros((num_timesteps, num_coins, 2), dtype=np.int64)
        
        for coin_idx, coin in enumerate(coins):
            close_col = f"{coin}_close"
            volume_col = f"{coin}_volume"
            
            if close_col not in df.columns or volume_col not in df.columns:
                raise ValueError(f"Missing {close_col} or {volume_col} in data")
            
            if coin not in thresholds:
                raise ValueError(f"No thresholds for {coin}")
            
            # Price channel: compute log returns
            prices = df[close_col]
            price_returns = np.log(prices / prices.shift(1))
            
            # Volume channel: compute log changes
            volumes = df[volume_col]
            volume_changes = np.log(volumes / volumes.shift(1))
            
            # Apply price thresholds
            tau_low_price, tau_high_price = thresholds[coin]['price']
            price_tokens = np.full(len(price_returns), np.nan)
            price_tokens[price_returns <= tau_low_price] = 0  # down
            price_tokens[(price_returns > tau_low_price) & (price_returns <= tau_high_price)] = 1  # steady
            price_tokens[price_returns > tau_high_price] = 2  # up
            
            # Apply volume thresholds
            tau_low_vol, tau_high_vol = thresholds[coin]['volume']
            volume_tokens = np.full(len(volume_changes), np.nan)
            volume_tokens[volume_changes <= tau_low_vol] = 0  # down
            volume_tokens[(volume_changes > tau_low_vol) & (volume_changes <= tau_high_vol)] = 1  # steady
            volume_tokens[volume_changes > tau_high_vol] = 2  # up
            
            # Drop first row (NaN from log returns) and store
            tokens_array[:, coin_idx, 0] = price_tokens[1:]
            tokens_array[:, coin_idx, 1] = volume_tokens[1:]
        
        # Verify no NaNs
        if np.isnan(tokens_array).any():
            raise ValueError("NaN values in tokenized data!")
        
        return tokens_array
    
    def _predict(self, model, tokens: np.ndarray, device, output_length: int) -> tuple:
        """
        Run autoregressive model inference
        
        Args:
            model: Trained model
            tokens: Token array, shape (input_length, num_coins, 2)
            device: Device to run on
            output_length: Number of hours to predict
            
        Returns:
            (predictions, probabilities) tuple
            predictions: shape (output_length,)
            probabilities: shape (output_length, 3)
        """
        # Create batch tensor (1, input_length, num_coins, 2)
        X = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Autoregressive generation
            preds = model.generate(X, max_len=output_length)  # (1, output_length)
            
            # For probabilities, we need to do forward pass with teacher forcing
            # or track probabilities during generation
            # For simplicity, we'll do a forward pass to get logits
            # Note: In true autoregressive inference, each step's probability would differ
            # This is an approximation
            logits = model(X, targets=None)  # Will use autoregressive mode
            probs = torch.softmax(logits, dim=-1) if logits is not None else None
            
            # If model.generate doesn't return logits, we can't get exact probabilities
            # For now, create uniform probabilities as placeholder
            if probs is None:
                probs = torch.ones(1, output_length, 3) / 3.0
        
        # Convert to numpy
        predictions = preds.cpu().numpy()[0]  # (output_length,)
        probabilities = probs.cpu().numpy()[0] if probs is not None else np.ones((output_length, 3)) / 3.0
        
        return predictions, probabilities
    
    def _format_results(self, predictions: np.ndarray, probabilities: np.ndarray,
                       timestamp: datetime, target_coin: str, output_length: int) -> Dict:
        """
        Format predictions as JSON output
        
        Args:
            predictions: Predicted tokens, shape (output_length,)
            probabilities: Prediction probabilities, shape (output_length, 3)
            timestamp: Current timestamp
            target_coin: Target coin symbol
            output_length: Number of prediction hours
            
        Returns:
            Formatted results dictionary
        """
        labels = {0: "down", 1: "steady", 2: "up"}
        
        predictions_list = []
        for h in range(output_length):
            pred_token = int(predictions[h])
            pred_label = labels[pred_token]
            confidence = float(probabilities[h, pred_token])
            
            predictions_list.append({
                "hour": h + 1,
                "timestamp": (timestamp + timedelta(hours=h+1)).isoformat(),
                "prediction": pred_label,
                "confidence": confidence,
                "probabilities": {
                    "down": float(probabilities[h, 0]),
                    "steady": float(probabilities[h, 1]),
                    "up": float(probabilities[h, 2])
                }
            })
        
        results = {
            "timestamp": timestamp.isoformat(),
            "coin": target_coin,
            "model_version": "v1.0.0",
            "horizon_hours": output_length,
            "predictions": predictions_list,
            "metadata": {
                "inference_time_ms": 0,  # Would measure in production
                "config": {
                    "input_length": self.config['sequences']['input_length'],
                    "output_length": output_length,
                    "num_coins": len(self.config['data']['coins'])
                }
            }
        }
        
        return results

