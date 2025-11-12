#!/usr/bin/env python3
"""
main.py — Parametrized trainer for DistilBERT on GLUE (default: MRPC)

Example:
  python main.py \
    --checkpoint_dir ./models \
    --model_name_or_path distilbert-base-uncased \
    --task_name mrpc \
    --learning_rate 2e-5 \
    --scheduler_name linear \
    --warmup_ratio 0.06 \
    --weight_decay 0.01 \
    --optimizer_name AdamW \
    --epochs 3 \
    --log_wandb \
    --wandb_project mrpc-tuning
"""

import os
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import datasets
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

# --------------------------
# DataModule
# --------------------------


class GLUEDataModule(L.LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True)

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str = "fit"):
        self.dataset = datasets.load_dataset("glue", self.task_name)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            cols = [
                c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=cols)
        self.eval_splits = [
            x for x in self.dataset.keys() if "validation" in x]

    def convert_to_features(self, example_batch, indices=None):
        # Encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_pairs = list(
                zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_pairs = example_batch[self.text_fields[0]]

        features = self.tokenizer.batch_encode_plus(
            texts_or_pairs, max_length=self.max_seq_length, padding="max_length", truncation=True
        )
        features["labels"] = example_batch["label"]
        return features

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"],
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        # e.g., MNLI matched/mismatched
        return [
            DataLoader(self.dataset[x], batch_size=self.eval_batch_size,
                       num_workers=self.num_workers, pin_memory=True)
            for x in self.eval_splits
        ]

# --------------------------
# LightningModule
# --------------------------


def make_scheduler(optimizer, name, num_warmup_steps, num_training_steps):
    name = str(name).lower()
    if name == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if name == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    raise ValueError(f"Unknown scheduler: {name}")


class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        scheduler_name: str = "linear",
        warmup_ratio: float = 0.06,
        optimizer_name: str = "AdamW",
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config)
        self.metric = evaluate.load("glue", task_name)
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(
            logits, axis=1) if self.hparams.num_labels > 1 else logits.squeeze()
        labels = batch["labels"]
        self.validation_step_outputs.append(
            {"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        preds = torch.cat(
            [x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat(
            [x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"]
                           for x in self.validation_step_outputs]).mean()
        metrics = self.metric.compute(predictions=preds, references=labels)
        # Prefix with val_
        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(val_metrics, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = self.hparams.optimizer_name.lower()
        opt_cls = torch.optim.AdamW if opt == "adamw" else torch.optim.Adam
        optimizer = opt_cls(
            param_groups,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * float(self.hparams.warmup_ratio))
        scheduler = make_scheduler(
            optimizer, self.hparams.scheduler_name, warmup_steps, total_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# --------------------------
# Utilities
# --------------------------


def default_monitor_for_task(task_name: str) -> str:
    """
    Choose a sensible validation metric to monitor per GLUE task.
    """
    task_name = task_name.lower()
    if task_name == "cola":
        return "val_matthews_correlation"
    if task_name == "stsb":
        # evaluate returns pearson and spearmanr; spearman is commonly used for early stopping
        return "val_spearmanr"
    # For most others (mrpc/qqp/sst2/mnli/qnli/rte/wnli), accuracy is present
    return "val_accuracy"

# --------------------------
# Main / CLI
# --------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train DistilBERT on a GLUE task with Lightning.")
    # Data / model
    parser.add_argument("--model_name_or_path", type=str,
                        default="distilbert-base-uncased")
    parser.add_argument(
        "--task_name",
        type=str,
        default="mrpc",
        choices=["cola", "sst2", "mrpc", "qqp", "stsb",
                 "mnli", "qnli", "rte", "wnli", "ax"],
    )
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # Optim / sched
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler_name", type=str,
                        default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--optimizer_name", type=str,
                        default="AdamW", choices=["AdamW", "Adam"])
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)

    # Training control
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="32-true",
                        help="Lightning precision, e.g. 32-true, 16-mixed")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")

    # Logging / checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--monitor_metric", type=str,
                        default=None, help="Override auto metric selection")
    parser.add_argument("--monitor_mode", type=str,
                        default="max", choices=["min", "max"])
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    # Weights & Biases
    parser.add_argument("--log_wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="mrpc-tuning")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    # Seeding for reproducibility
    L.seed_everything(args.seed)

    # Data
    dm = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    dm.setup("fit")

    # Model
    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=dm.num_labels,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler_name,
        warmup_ratio=args.warmup_ratio,
        optimizer_name=args.optimizer_name,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
    )

    # Logging
    logger = False
    if args.log_wandb:
        run_name = args.wandb_run_name
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            log_model=False,
        )
        # snapshot of important hparams
        try:
            logger.experiment.config.update(vars(args), allow_val_change=True)
        except Exception:
            pass

    # Checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    monitor_metric = args.monitor_metric or default_monitor_for_task(
        args.task_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="{epoch}-{" + monitor_metric + ":.4f}",
        monitor=monitor_metric,
        mode=args.monitor_mode,
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
    )

    # Train
    trainer.fit(model, datamodule=dm)

    # Print best
    if ckpt_cb.best_model_path:
        print(f"[OK] Best checkpoint: {ckpt_cb.best_model_path}")
        print(
            f"[OK] Best score ({monitor_metric}): {ckpt_cb.best_model_score}")
    else:
        print("[WARN] No best checkpoint recorded.")

    # Finish W&B cleanly if enabled
    if args.log_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
