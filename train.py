import os.path
import pickle

import numpy as np
import sklearn
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import Dataset, Subset
from transformers import TrainingArguments, IntervalStrategy, Trainer, AutoTokenizer, PreTrainedTokenizerFast, \
    PreTrainedModel, AutoModelForSequenceClassification, Adafactor, EvalPrediction
from transformers.integrations import TensorBoardCallback
from transformers.optimization import AdafactorSchedule

from config import Config
from process_data import FeministDataset, FeminismDetectionDataset

eval_steps: int = 150
batch_size: int = 1


class FeminismTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(FeminismTrainer, self).__init__(*args, **kwargs)
        self.loss_fn: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, model: PreTrainedModel, inputs: dict, return_outputs: bool = False):
        logits, labels = model(**inputs)
        loss: torch.Tensor = self.loss_fn(logits, labels)
        return (loss, (loss, logits)) if return_outputs else loss


def compute_metrics(ep: EvalPrediction) -> dict[str, float]:
    logits, labels = ep
    predictions: np.ndarray = np.argmax(logits, axis=-1)
    true: np.ndarray = np.argmax(labels, axis=-1)
    report_dict: dict = sklearn.metrics.classification_report(true, predictions, output_dict=True)
    return report_dict


def train_classifier(eval_only: bool = False) -> None:
    # model_name: str = "deepset/gelectra-large"
    # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)
    # model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_name).to(Config.device)
    # dataset: FeminismDetectionDataset = FeminismDetectionDataset(
    #     tokenizer=tokenizer, max_length=model.config.max_position_embeddings)
    dataset: FeminismDetectionDataset = FeminismDetectionDataset()
    dataset_size: int = len(dataset)
    indices: list[int] = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    split: int = int(np.floor(0.1 * dataset_size))  # 0.05
    train_indices, val_indices = indices[split:], indices[:split]
    val_dataset: Dataset = Subset(dataset, val_indices)
    train_dataset: Dataset = Subset(dataset, train_indices)
    count_vect: CountVectorizer = CountVectorizer()
    X_train_counts = count_vect.fit_transform([x.text for x in train_dataset])
    tfidf_transformer: TfidfTransformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf: MultinomialNB = MultinomialNB().fit(X_train_tfidf, [x.label for x in train_dataset])
    docs_new: list[str] = [x.text for x in val_dataset]
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    y_true: list[int] = [x.label for x in val_dataset]
    y_pred: list[int] = list(clf.predict(X_new_tfidf))
    probabilities: list[tuple[float, float]] = clf.predict_proba(X_new_tfidf)
    texts_with_proba: list[tuple[str, float]] = [(docs_new[i], probabilities[i][1]) for i in range(len(docs_new))]
    texts_with_proba.sort(key=lambda x: x[1])
    print(sklearn.metrics.classification_report(y_true, y_pred))
    # use_cpu: bool = Config.device == torch.device("cpu")
    # train_args: TrainingArguments = TrainingArguments(
    #     output_dir=Config.output_dir, dataloader_pin_memory=False, logging_steps=eval_steps,
    #     logging_strategy=IntervalStrategy.STEPS, evaluation_strategy=IntervalStrategy.STEPS, eval_steps=eval_steps,
    #     per_device_train_batch_size=batch_size, per_device_eval_batch_size=1, num_train_epochs=1,
    #     save_strategy=IntervalStrategy.STEPS, gradient_accumulation_steps=1,  # learning_rate=1e-3, adafactor=True,
    #     seed=42, save_steps=eval_steps, gradient_checkpointing=False, save_total_limit=2)  # fp16=not use_cpu,
    #
    # optimizer = Adafactor(model.parameters(), lr=2e-3, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8,
    #                       beta1=None, weight_decay=0.0, relative_step=False, scale_parameter=False, warmup_init=False)
    # lr_scheduler = AdafactorSchedule(optimizer, initial_lr=2e-3)
    # trainer: Trainer = Trainer(
    #     model=model, args=train_args, train_dataset=train_dataset, callbacks=[TensorBoardCallback],
    #     eval_dataset=val_dataset, optimizers=(optimizer, lr_scheduler), compute_metrics=compute_metrics)
    # if eval_only:
    #     trainer.evaluate(eval_dataset=val_dataset)
    # else:
    #     trainer.train()
    # trainer.save_model(Config.models_dir)


def train_clm(eval_only: bool = False) -> None:
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
        seed=42, save_steps=eval_steps, fp16=not use_cpu, gradient_checkpointing=False, save_total_limit=2)  # 4e-2
    val_dataset: Dataset = Subset(dataset, val_indices)
    trainer: Trainer = Trainer(
        model=Config.model(), args=train_args, train_dataset=Subset(dataset, train_indices),
        callbacks=[TensorBoardCallback], eval_dataset=val_dataset)
    if eval_only:
        trainer.evaluate(eval_dataset=val_dataset)
    else:
        trainer.train()
    trainer.save_model(Config.models_dir)


def train_sklearn():
    train_dataset: FeminismDetectionDataset = FeminismDetectionDataset()
    X_train_counts = Config.count_vect.fit_transform([x.text for x in train_dataset])
    X_train_tfidf = Config.tfidf_transformer.fit_transform(X_train_counts)
    Config.clf.fit(X_train_tfidf, [x.label for x in train_dataset])


if __name__ == "__main__":
    # train_clm()
    train_classifier()
    # train_sklearn()
    pass
