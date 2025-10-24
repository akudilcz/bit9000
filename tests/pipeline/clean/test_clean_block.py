import pandas as pd
import numpy as np

from src.pipeline.step_02_clean.clean_block import CleanBlock
from src.pipeline.step_01_download.download_block import DownloadBlock


def test_clean_block_produces_artifact(monkeypatch, test_config, artifact_io):
    # Reuse stub collector from download block test
    from tests.pipeline.download.test_download_block import _StubCollector
    monkeypatch.setattr(
        "src.pipeline.step_01_download.download_block.DataCollector",
        _StubCollector,
    )

    # First run download to create a RawDataArtifact
    dl = DownloadBlock(test_config, artifact_io)
    raw_artifact = dl.run()

    # Run clean block on raw artifact
    cl = CleanBlock(test_config, artifact_io)
    clean_artifact = cl.run(raw_artifact)

    assert clean_artifact.num_timesteps == 5
    assert bool(clean_artifact.quality_metrics["index_monotonic"]) is True


def test_clean_block_with_duplicates_and_nans(monkeypatch, test_config, artifact_io):
    # Create data with duplicates and NaNs to test cleaning logic
    from tests.pipeline.download.test_download_block import _StubCollector
    monkeypatch.setattr(
        "src.pipeline.step_01_download.download_block.DataCollector",
        _StubCollector,
    )
    
    # First get raw data
    dl = DownloadBlock(test_config, artifact_io)
    raw_artifact = dl.run()
    
    # Manually modify the saved data to add duplicates and NaNs
    import pandas as pd
    df = artifact_io.read_dataframe(raw_artifact.path)
    
    # Add duplicates by duplicating a row
    df_with_dups = pd.concat([df, df.iloc[[0]]])  # Duplicate first row
    df_with_dups.iloc[0, 0] = np.nan  # Add a NaN
    
    # Write modified data back
    modified_path = artifact_io.write_dataframe(df_with_dups, "download", "raw_data_modified")
    raw_artifact.path = modified_path
    
    # Now run clean block
    cl = CleanBlock(test_config, artifact_io)
    clean_artifact = cl.run(raw_artifact)
    
    # Should have cleaned the data
    assert clean_artifact.quality_metrics["duplicate_rows_removed"] >= 1
    assert clean_artifact.quality_metrics["remaining_nans"] == 0

