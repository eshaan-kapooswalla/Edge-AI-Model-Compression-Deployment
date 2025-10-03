#!/usr/bin/env bash
set -euo pipefail

scripts/train_baseline.sh
scripts/prune.sh
scripts/quantize.sh
scripts/distill.sh
