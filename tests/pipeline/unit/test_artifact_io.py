import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.pipeline.io import ArtifactIO
from src.pipeline.schemas import ArtifactMetadata


def test_compute_hash_various_types(tmp_path):
    io = ArtifactIO(base_dir=str(tmp_path / "test_artifacts"))
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [1, 3]})
    h1 = io.compute_hash(df1)
    h2 = io.compute_hash(df2)
    assert isinstance(h1, str) and len(h1) == 32
    assert h1 != h2

    h_arr = io.compute_hash(np.array([1, 2, 3]))
    assert isinstance(h_arr, str) and len(h_arr) == 32

    h_str = io.compute_hash("hello")
    assert isinstance(h_str, str) and len(h_str) == 32

    h_other = io.compute_hash({"k": 1})
    assert isinstance(h_other, str) and len(h_other) == 32


def test_write_dataframe_simple(tmp_path):
    io = ArtifactIO(base_dir=str(tmp_path / "artifacts"))

    df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="h"))
    data_path = io.write_dataframe(df, block_name="download", artifact_name="raw_data")

    assert data_path.exists()
    assert data_path.name == "raw_data.parquet"


def test_array_and_json_roundtrip(tmp_path):
    io = ArtifactIO(base_dir=str(tmp_path / "artifacts"))
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    apath = io.write_array(arr, block_name="seq", artifact_name="X")
    assert io.read_array(apath).shape == (2, 2)
    assert apath.name == "X.npz"

    meta = ArtifactMetadata(schema_name="meta")
    jpath = io.write_json({"hello": "world"}, block_name="meta", artifact_name="info", metadata=meta)
    data = io.read_json(jpath)
    assert data["hello"] == "world"
    assert jpath.name == "info.json"


def test_write_model_and_read_model(tmp_path):
    io = ArtifactIO(base_dir=str(tmp_path / "artifacts"))
    state = {"weights": [1, 2, 3]}
    mpath = io.write_model(state, block_name="train", artifact_name="model")
    loaded = io.read_model(mpath, device="cpu")
    assert loaded["weights"] == [1, 2, 3]
    assert mpath.name == "model.pt"


def test_read_dataframe_read_array_read_model_read_json(tmp_path):
    io = ArtifactIO(base_dir=str(tmp_path / "artifacts"))
    # Test read_dataframe
    df = pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2024-01-01", periods=2, freq="h"))
    df_path = io.write_dataframe(df, block_name="test", artifact_name="df")
    loaded_df = io.read_dataframe(df_path)
    pd.testing.assert_frame_equal(df, loaded_df, check_freq=False)

    # Test read_array
    arr = np.array([[1, 2], [3, 4]])
    arr_path = io.write_array(arr, block_name="test", artifact_name="arr")
    loaded_arr = io.read_array(arr_path)
    np.testing.assert_array_equal(arr, loaded_arr)

    # Test read_model
    model_state = {"weights": [1, 2, 3]}
    model_path = io.write_model(model_state, block_name="test", artifact_name="model")
    loaded_model = io.read_model(model_path, device="cpu")
    assert loaded_model["weights"] == [1, 2, 3]

    # Test read_json
    json_data = {"key": "value", "nested": {"a": 1}}
    json_path = io.write_json(json_data, block_name="test", artifact_name="data")
    loaded_json = io.read_json(json_path)
    assert loaded_json["key"] == "value"
    assert loaded_json["nested"]["a"] == 1


