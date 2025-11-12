# DistilBERT GLUE Fine-Tuning (MRPC Task)

This project fine-tunes **DistilBERT** on the [GLUE benchmark](https://gluebenchmark.com/),  
using **PyTorch Lightning**, **Hugging Face Transformers**, and **Weights & Biases** for logging.  
It supports local runs and Dockerized training with GPU or CPU backends.

---

## Quick Start (Local)

### Setup environment

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run training

Default: MRPC task, 3 epochs, DistilBERT base, linear LR schedule.

```bash
python main.py   --task_name mrpc   --epochs 3   --devices 1 --accelerator auto
```

### Enable Weights & Biases logging

```bash
export WANDB_API_KEY=<your-key>
python main.py   --task_name mrpc   --epochs 3   --learning_rate 2e-5   --scheduler_name cosine   --weight_decay 0.01   --warmup_ratio 0.06   --optimizer_name AdamW   --train_batch_size 32   --eval_batch_size 32   --log_wandb   --wandb_project mrpc-tuning   --wandb_run_name mrpc_best_cosine_lr2e-5
```

> 💡 Best validation accuracy achieved: **84.8%**  
> using `lr=2e-5`, `scheduler=cosine`, `weight_decay=0.01`, `warmup_ratio=0.06`, `optimizer=AdamW`, batch size `32`.

---

## Run with Docker

### Build (GPU version, default)

```bash
docker build -t mrpc-trainer:cuda .
```

### Run training (GPU)

```powershell
docker run --rm -it --gpus all \
--ipc=host --shm-size=1g \
-v "$PWD/checkpoints:/app/checkpoints" \
-v "$PWD/.cache/hf:/cache/hf" \
-v "$PWD/.cache/wandb:/cache/wandb" \
-v "$PWD/.cache/tmp:/cache/tmp" \
-e TMPDIR=/cache/tmp \
-e TOKENIZERS_PARALLELISM=false \
-e WANDB_API_KEY=efdXXXXXXXXXXXXXX \
-e WANDB_PROJECT=mrpc-tuning \
mrpc-trainer:cuda \
python main.py \
--task_name mrpc \
--learning_rate 2e-5 \
--scheduler_name cosine \
--weight_decay 0.01 \
--warmup_ratio 0.06 \
--optimizer_name AdamW \
--epochs 3 \
--devices 1 --accelerator gpu \
--checkpoint_dir /app/checkpoints \
--num_workers 0 \
--log_wandb
```

### Build (CPU-only)

```bash
docker build --build-arg BASE_IMAGE=python:3.10-slim -t mrpc-trainer:cpu .
```

### Run (CPU)

```bash
docker run --rm -it   -v $PWD/checkpoints:/app/checkpoints   mrpc-trainer:cpu   python main.py --accelerator cpu --devices 1 --epochs 3
```

---

## Weights & Biases Logging

If `--log_wandb` is set, the script logs:

| Metric                                 | Description                             |
| -------------------------------------- | --------------------------------------- |
| `train_loss_step` / `train_loss_epoch` | Training loss curves                    |
| `val_loss`                             | Validation loss per epoch               |
| `val_accuracy`, `val_f1`, etc.         | Task-specific GLUE metrics              |
| `lr-AdamW/group0`                      | Learning rate schedule                  |
| System metrics                         | GPU utilization, memory, CPU load, etc. |
| Config panel                           | All hyperparameters used in the run     |

Example dashboards include **loss vs. step**, **validation accuracy vs. epoch**, and **learning rate curves**.

---

## Project Structure

```
.
├── main.py                # Entry point (Lightning + Transformers)
├── requirements.txt       # Dependencies
├── Dockerfile             # Container build definition
├── README.md              # You are here
├── checkpoints/           # Saved models (created at runtime)

```

---

## CLI Arguments (excerpt)

| Argument             | Default         | Description                                    |
| -------------------- | --------------- | ---------------------------------------------- |
| `--task_name`        | `mrpc`          | GLUE task name                                 |
| `--learning_rate`    | `2e-5`          | Base learning rate                             |
| `--scheduler_name`   | `linear`        | LR schedule: `linear`, `cosine`, or `constant` |
| `--weight_decay`     | `0.01`          | Weight decay                                   |
| `--warmup_ratio`     | `0.06`          | Warmup fraction of total steps                 |
| `--optimizer_name`   | `AdamW`         | Optimizer (`AdamW` or `Adam`)                  |
| `--epochs`           | `3`             | Number of epochs                               |
| `--train_batch_size` | `32`            | Training batch size                            |
| `--eval_batch_size`  | `32`            | Eval batch size                                |
| `--log_wandb`        | _(flag)_        | Enable W&B logging                             |
| `--checkpoint_dir`   | `./checkpoints` | Where checkpoints are saved                    |

For full list:

```bash
python main.py --help
```

---
