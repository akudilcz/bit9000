#!/usr/bin/env python3
"""Fix failing tests by updating patches"""

import re

# Fix evaluate tests
with open('tests/pipeline/evaluate/test_evaluate_block.py', 'r') as f:
    content = f.read()

# Replace all occurrences of the old patch
content = content.replace(
    "patch('src.pipeline.step_08_evaluate.evaluate_block.CryptoTransformerV4')",
    "patch('src.model.create_model')"
)

# Also need to update the mock usage
content = content.replace(
    "mock_model_class",
    "mock_create_model"
)

with open('tests/pipeline/evaluate/test_evaluate_block.py', 'w') as f:
    f.write(content)

print("Fixed evaluate tests")

# Fix inference tests
with open('tests/pipeline/inference/test_inference_block.py', 'r') as f:
    content = f.read()

content = content.replace(
    "patch('src.pipeline.step_09_inference.inference_block.CryptoTransformerV4')",
    "patch('src.model.create_model')"
)

content = content.replace(
    "mock_model_class",
    "mock_create_model"
)

with open('tests/pipeline/inference/test_inference_block.py', 'w') as f:
    f.write(content)

print("Fixed inference tests")

# Fix integration tests
with open('tests/pipeline/integration/test_full_pipeline.py', 'r') as f:
    content = f.read()

# Fix the DataCollector patch
content = content.replace(
    "patch('src.pipeline.step_01_download.data_collector.CryptoDataCollector')",
    "patch('src.pipeline.step_01_download.data_collector.DataCollector')"
)

content = content.replace(
    "mock_collector_class",
    "mock_collector"
)

# Fix the Trainer patch
content = content.replace(
    "patch('src.pipeline.step_07_train.train_block.Trainer')",
    "patch('torch.load')"
)

with open('tests/pipeline/integration/test_full_pipeline.py', 'w') as f:
    f.write(content)

print("Fixed integration tests")

print("All test fixes applied!")

