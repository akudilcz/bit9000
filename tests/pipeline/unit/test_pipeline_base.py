from src.pipeline.base import PipelineBlock
from src.pipeline.io import ArtifactIO


class _DummyBlock(PipelineBlock):
    def run(self, **kwargs):
        return None


def test_create_metadata_schema_name(test_config, artifact_io):
    blk = _DummyBlock(test_config, artifact_io)
    meta = blk.create_metadata({"up": "stream"})
    assert meta.schema_name == "_dummy"
    assert meta.upstream_inputs["up"] == "stream"


def test_block_name_derivation(test_config, artifact_io):
    # Test that block_name is derived from class name
    blk = _DummyBlock(test_config, artifact_io)
    assert blk.block_name == "_dummy"


