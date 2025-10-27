import pandas as pd
from datetime import datetime

from src.pipeline.step_01_download.download_block import DownloadBlock


class _StubCollector:
    def __init__(self, config):
        self.config = config

    def collect_all_data(self, start_date, end_date):
        idx = pd.date_range(datetime(2024, 1, 1), periods=5, freq="h")
        df = pd.DataFrame({
            "BTC_open": [1, 2, 3, 4, 5],
            "BTC_high": [1, 2, 3, 4, 5],
            "BTC_low": [1, 2, 3, 4, 5],
            "BTC_close": [1, 2, 3, 4, 6],
            "BTC_volume": [10, 10, 10, 10, 10],
            "ETH_open": [1, 2, 3, 4, 5],
            "ETH_high": [1, 2, 3, 4, 5],
            "ETH_low": [1, 2, 3, 4, 5],
            "ETH_close": [1, 2, 3, 4, 6],
            "ETH_volume": [10, 10, 10, 10, 10],
        }, index=idx)
        return df


def test_download_block_creates_artifacts(monkeypatch, test_config, artifact_io):
    # Swap out DataCollector with stub to avoid network/IO
    monkeypatch.setattr(
        "src.pipeline.step_01_download.download_block.DataCollector",
        _StubCollector,
    )

    block = DownloadBlock(test_config, artifact_io)
    artifact = block.run()

    # Validate artifact fields
    assert artifact.num_timesteps == 5
    assert artifact.num_coins == 2
    assert any(c.endswith("_close") for c in artifact.columns)

    # Artifact file should exist
    assert artifact.path.exists()
    assert artifact.path.name == "raw_data.parquet"



