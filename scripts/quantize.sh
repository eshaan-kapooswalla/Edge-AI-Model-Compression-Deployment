#!/usr/bin/env bash
set -euo pipefail

python -m src.model_compression.cli quantize --checkpoint models/baseline_resnet18.pt --output models/quantized_dynamic.pt
