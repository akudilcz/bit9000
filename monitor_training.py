#!/usr/bin/env python3
"""Monitor training progress by tailing the log file"""

import subprocess
import sys
import time

log_file = '/home/andrew/bit9000/train_output.log'

print(f"Monitoring {log_file}...")
print("=" * 80)
print("Training will show:")
print("  - Epoch N/200")
print("  - train_loss, train_acc (overall accuracy across 4 horizons)")
print("  - val_loss, val_acc")
print("  - buy_precision (% of BUY signals that were correct)")
print("=" * 80)
print()

try:
    # Follow the log file in real-time
    process = subprocess.Popen(['tail', '-f', log_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        if 'Epoch' in line or 'BUY Precision' in line or 'COMPLETE' in line or 'Early stopping' in line:
            print(line.rstrip())
except KeyboardInterrupt:
    print("\nStopped monitoring")
except FileNotFoundError:
    print(f"Log file not found yet: {log_file}")
    print("Training may not have started yet")
