"""Pytest configuration and shared fixtures"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml


@pytest.fixture
def test_config(tmp_path):
    """Load test config and override paths to use tmp_path"""
    config_path = Path(__file__).parent.parent / "test_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override all paths to use tmp_path for isolation
    test_artifacts = tmp_path / "test_artifacts"
    test_artifacts.mkdir(exist_ok=True)
    
    config['artifacts_base_dir'] = str(test_artifacts)
    config['data']['data_dir'] = str(test_artifacts / 'step_01_download')
    config['data']['processed_dir'] = str(test_artifacts / 'step_02_clean')
    config['model']['checkpoint_dir'] = str(test_artifacts / 'step_07_train' / 'checkpoints')
    config['evaluation']['results_dir'] = str(test_artifacts / 'step_08_evaluate')
    config['backtest']['results_dir'] = str(test_artifacts / 'step_10_backtest')
    config['visualization']['export_dir'] = str(test_artifacts / 'step_11_visualize')
    config['logging']['file'] = None  # No file logging during tests
    
    return config


@pytest.fixture
def canned_small_data():
    """Generate small clean dataset: 3 coins × 60 days hourly"""
    start = datetime(2024, 1, 1)
    hours = 60 * 24  # 60 days
    timestamps = pd.date_range(start, periods=hours, freq='h')
    
    data = {}
    for coin in ['BTC', 'ETH', 'SOL']:
        # Generate realistic-looking price data
        base_price = {'BTC': 40000, 'ETH': 2500, 'SOL': 100}[coin]
        volatility = 0.02
        
        # Random walk
        returns = np.random.normal(0, volatility, hours)
        log_prices = np.cumsum(returns)
        prices = base_price * np.exp(log_prices)
        
        data[f'{coin}_open'] = prices
        data[f'{coin}_high'] = prices * (1 + np.abs(np.random.normal(0, 0.01, hours)))
        data[f'{coin}_low'] = prices * (1 - np.abs(np.random.normal(0, 0.01, hours)))
        data[f'{coin}_close'] = prices
        data[f'{coin}_volume'] = np.random.uniform(1e6, 1e8, hours)
    
    df = pd.DataFrame(data, index=timestamps)
    return df


@pytest.fixture
def canned_anomalies_data():
    """Generate dataset with quality issues: NaNs, gaps, duplicates"""
    start = datetime(2024, 1, 1)
    hours = 30 * 24  # 30 days
    timestamps = pd.date_range(start, periods=hours, freq='h')
    
    data = {}
    for coin in ['BTC', 'ETH']:
        base_price = {'BTC': 40000, 'ETH': 2500}[coin]
        prices = base_price * np.exp(np.random.normal(0, 0.02, hours).cumsum())
        
        data[f'{coin}_open'] = prices
        data[f'{coin}_high'] = prices * 1.01
        data[f'{coin}_low'] = prices * 0.99
        data[f'{coin}_close'] = prices
        data[f'{coin}_volume'] = np.random.uniform(1e6, 1e8, hours)
    
    df = pd.DataFrame(data, index=timestamps)
    
    # Introduce anomalies
    # 1. NaNs
    df.iloc[100:105] = np.nan
    df.iloc[200, 0] = np.nan
    
    # 2. Duplicates
    df = pd.concat([df, df.iloc[50:55]])
    
    # 3. Out-of-order (will create gaps when sorted)
    df = df.sample(frac=1.0)
    
    return df


@pytest.fixture
def canned_leak_trap_data():
    """
    Generate dataset designed to detect leakage.
    Price at t+1 is exactly price[t] + 1 if feature[t] > 0, else price[t] - 1.
    If leakage exists, model will achieve 100% accuracy.
    """
    start = datetime(2024, 1, 1)
    hours = 30 * 24
    timestamps = pd.date_range(start, periods=hours, freq='h')
    
    # Create deterministic price series
    base = 10000.0
    prices = [base]
    indicator = np.random.choice([-1, 1], hours - 1)
    
    for ind in indicator:
        prices.append(prices[-1] + ind * 10)
    
    prices = np.array(prices)
    
    data = {}
    for coin in ['BTC']:
        data[f'{coin}_open'] = prices
        data[f'{coin}_high'] = prices * 1.001
        data[f'{coin}_low'] = prices * 0.999
        data[f'{coin}_close'] = prices
        data[f'{coin}_volume'] = np.ones(hours) * 1e7
    
    df = pd.DataFrame(data, index=timestamps)
    return df


@pytest.fixture
def artifacts_dir(tmp_path):
    """Temporary artifacts directory"""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return artifacts


@pytest.fixture
def artifact_io(test_config):
    """ArtifactIO instance with test artifacts directory"""
    from src.pipeline.io import ArtifactIO
    return ArtifactIO(base_dir=test_config['artifacts_base_dir'])

