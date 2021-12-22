import os.path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import TrainingArguments, IntervalStrategy, Trainer
from transformers.integrations import TensorBoardCallback

from config import Config
from data import FeministDataset

eval_steps: int = 50
batch_size: int = 1


def train(eval_only: bool = False) -> None:
    dataset: Dataset = FeministDataset()
    dataset_size: int = len(dataset)
    indices: list[int] = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    split: int = int(np.floor(0.0005 * dataset_size))  # 0.05
    train_indices, val_indices = indices[split:], indices[:split]
    use_cpu: bool = Config.device == torch.device("cpu")
    train_args: TrainingArguments = TrainingArguments(
        output_dir=Config.output_dir, dataloader_pin_memory=False, logging_steps=eval_steps,
        logging_strategy=IntervalStrategy.STEPS, evaluation_strategy=IntervalStrategy.STEPS, eval_steps=eval_steps,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=1, num_train_epochs=1,
        save_strategy=IntervalStrategy.STEPS, learning_rate=4e-3, adafactor=True, gradient_accumulation_steps=1,
        seed=42, save_steps=10, fp16=not use_cpu, gradient_checkpointing=False, save_total_limit=2)  # 4e-2
    val_dataset: Dataset = Subset(dataset, val_indices)
    trainer: Trainer = Trainer(
        model=Config.model(), args=train_args, train_dataset=Subset(dataset, train_indices),
        callbacks=[TensorBoardCallback], eval_dataset=val_dataset)
    if eval_only:
        trainer.evaluate(eval_dataset=val_dataset)
    else:
        trainer.train()
    torch.save(Config.model().state_dict(), os.path.join(Config.models_dir, "model.pt"))


train()
