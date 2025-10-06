<div align="center">

# Model Compression for On‑Device Deployment (CIFAR‑10)

Smaller, faster, and accurate deep learning models for edge devices. This project implements and benchmarks three core compression strategies — pruning, quantization, and knowledge distillation — on CIFAR‑10 using TensorFlow/Keras, with a clean MLOps‑style workflow.

</div>

## Highlights
- End‑to‑end baseline: ResNet50 with in‑model preprocessing and data augmentation
- Reproducible env: pinned `requirements.txt`, virtualenv, deterministic scripts
- Compression toolchain:
  - Pruning (TF‑MOT)
  - Post‑training quantization (dynamic/full‑integer, TFLite)
  - Knowledge distillation (custom `keras.Model` subclass)
- Unified benchmarking: size, latency, peak RAM, and accuracy, logged to Markdown

## Repository Structure
- `train_baseline.py` — Train and save the baseline ResNet50 model
- `benchmark.py` — Benchmark models (SavedModel or `.tflite`) and log results
- `prune_model.py` — Load baseline and prep for pruning (TF‑MOT)
- `quantize_model.py` — Create dynamic and full‑integer quantized TFLite models
- `distill_model.py` — Train a compact student via knowledge distillation and save it
- `benchmark_results.md` — Rolling log of benchmark runs as a Markdown table
- `models/` — Saved models and artifacts (ignored by git)
- `src/` `scripts/` `notebooks/` — Library code, automation, and analyses

## Setup
```bash
python3 -m venv venv            # or .venv
source venv/bin/activate        # macOS/Linux
pip install -U pip
pip install -r requirements.txt
```

## Baseline: Train ResNet50
```bash
source venv/bin/activate
python train_baseline.py
```
This will:
- Load CIFAR‑10
- Apply normalization + on‑the‑fly augmentation
- Build ResNet50 with a global‑avg‑pool + dense(10) head
- Compile with Adam + SparseCategoricalCrossentropy
- Print summary and train; then save to `models/baseline_model/` (SavedModel)

## Benchmarking
Benchmark any model (SavedModel dir or `.tflite`) for:
- Size (MB)
- Average single‑image latency (ms)
- Peak RAM (MiB)
- Accuracy on CIFAR‑10 test set

Configure the target in `benchmark.py` via `MODEL_PATH`, then run:
```bash
source venv/bin/activate
python benchmark.py
```
Results append to `benchmark_results.md`.

## Pruning (TF‑MOT)
```bash
source venv/bin/activate
python prune_model.py
```
Loads the baseline model and prepares it for TensorFlow Model Optimization Toolkit pruning workflows. Extend this script to wrap layers with `tfmot.sparsity.keras.prune_low_magnitude`, fine‑tune, then strip pruning wrappers before saving.

## Quantization (TFLite)
```bash
source venv/bin/activate
python quantize_model.py
```
Generates:
- Dynamic range quantized model: `models/quantized_dynamic_range.tflite`
- Full integer quantized model: `models/quantized_integer_only.tflite`

Benchmark both by pointing `MODEL_PATH` in `benchmark.py` to the respective file.

## Knowledge Distillation
```bash
source venv/bin/activate
python distill_model.py
```
Implements a custom `Distiller(keras.Model)` with overridden `train_step` to blend student loss and distillation loss (KLD over softened logits). Trains a compact CNN student and saves it to `models/student_model/` (and/or `models/distilled_student_model/`).

## Typical Workflow
1) Train baseline → `models/baseline_model/`
2) Quantize/prune/distill → produce artifacts under `models/`
3) Benchmark each → append to `benchmark_results.md`
4) Compare trade‑offs across size, speed, RAM, and accuracy

## Example Commands
```bash
# Baseline
python train_baseline.py && python benchmark.py

# Quantized (dynamic)
sed -i '' 's|MODEL_PATH = ".*"|MODEL_PATH = "models/quantized_dynamic_range.tflite"|' benchmark.py
python benchmark.py

# Full‑integer TFLite
sed -i '' 's|MODEL_PATH = ".*"|MODEL_PATH = "models/quantized_integer_only.tflite"|' benchmark.py
python benchmark.py

# Distilled student
sed -i '' 's|MODEL_PATH = ".*"|MODEL_PATH = "models/distilled_student_model"|' benchmark.py
python benchmark.py
```

## Reproducibility & Notes
- Environments: pinned in `requirements.txt`
- Artifacts: models under `models/` are ignored by git to keep the repo lean
- Benchmarks: durable record in `benchmark_results.md`

## License
MIT — see `LICENSE` if present. Otherwise, adapt as needed.

## Acknowledgements
- TensorFlow / Keras for training & SavedModel
- TensorFlow Model Optimization Toolkit (TF‑MOT) for pruning
- TFLite for deployment‑oriented quantization
