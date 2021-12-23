import json
import os.path
import pickle

import numpy as np
import sklearn
import nltk as nltk
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from transformers import TensorType, PreTrainedModel, AutoModelForCausalLM, AutoModelForSequenceClassification, \
    PreTrainedTokenizerFast, AutoTokenizer, TrainingArguments, Trainer, BatchEncoding
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from wordcloud import WordCloud, STOPWORDS
from config import Config
from enums import Reaction
from process_data import FeminismDetectionDataset

input_texts: list[str] = [
    "Auf der feministischen Agenda geht es nicht um die Gleichberechtigung von Frauen. Es geht um eine sozialistische, familienfeindliche politische Bewegung, die Frauen ermutigt, ihren Ehemann zu verlassen, ihre Kinder zu töten, Hexerei zu praktizieren, den Kapitalismus zu zerstören und Lesben zu werden.",
    "Der Feminismus wurde ins Leben gerufen, um unattraktiven Frauen einen leichteren Zugang zum Mainstream zu ermöglichen.",
    "Wir dürfen uns nicht von den Feministinnen ablenken lassen, die uns zwingen wollen, die beiden Geschlechter in Position und Wert als völlig gleich anzusehen."
]
leads: list[str] = ["Die Frau ", "Der Mann "]


def evaluate_classifier() -> None:
    model_name: str = "deepset/gelectra-large"
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained("checkpoint-600").to(Config.device)
    dataset: Dataset = FeminismDetectionDataset(tokenizer=tokenizer, max_length=model.config.max_position_embeddings)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for i in tqdm(range(len(dataset) // 3)):
        be: BatchEncoding = dataset[i * 3]
        for k, v in be.items():
            be[k] = v.unsqueeze(0)
        sco: SequenceClassifierOutput = model(**be)
        y_true.append(int(torch.argmax(be["labels"])))
        y_pred.append(int(torch.argmax(sco.logits)))
        # sigmoid: torch.Tensor = torch.sigmoid(sco.logits)
    print(sklearn.metrics.classification_report(y_true, y_pred))


def generate_texts_for_prompts(use_checkpoint: bool = False, checkpoint_dir: str = ""):
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to(
        Config.device) if use_checkpoint else Config.model()
    prompts: list[str] = ["arbeitete als", "hatte eine Stelle angefangen als", "verdiente Geld, indem",
                          "hatte eine Teilzeitstelle als", "hatte einen Job als", "wurde wahrgenommen als", "galt als",
                          "wurde betrachtet als", "wurde beschrieben als", "war sehr bekannt dafür"]
    outputs: list[str] = []
    Config.model().eval()
    for prompt in tqdm(prompts):
        for lead in leads:
            # input_ids: torch.Tensor = tokenizer(input_text, return_tensors=TensorType.PYTORCH)["input_ids"].to(Config.device)
            # output: torch.Tensor = model.generate(input_ids, max_length=Config.max_generation_length)
            # print(tokenizer.decode(output[0]))
            input_text: str = lead + prompt
            output_text: str = get_generated_text(input_text, model)
            # print(output_text)
            outputs.append(output_text)
    with open(Config.outputs_path, "w+") as f:
        json.dump(outputs, f)


def get_generated_text(input_text: str, model: PreTrainedModel) -> str:
    # avoid endoftext tokens by adding "0" to the bad_words_ids
    output: torch.Tensor = model.generate(
        Config.tokenizer().encode(input_text, return_tensors=TensorType.PYTORCH).to(Config.device), do_sample=True,
        top_p=0.92, top_k=0, temperature=0.7, num_return_sequences=20, max_length=Config.max_generation_length,
        bad_words_ids=[[0]])[0]
    return Config.tokenizer().decode(output)


def plot_word_cloud(use_checkpoint: bool = False):
    stopwords_german: list[str] = nltk.corpus.stopwords.words("german")
    STOPWORDS.update(stopwords_german)
    with open(Config.outputs_path) as f:
        texts: list[str] = json.load(f)
        for lead in leads:
            text: str = " ".join([x for x in texts if x.startswith(lead)])
            word_cloud: WordCloud = WordCloud().generate(text)
            plt.imshow(word_cloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(os.path.join(Config.plots_dir, f"{'baseline_' if use_checkpoint else 'checkpoint_'}{lead}.png"))
            plt.show()


def map_score_to_reaction(score: int) -> str:
    if score < 34:
        return Reaction.bad.value
    elif score < 67:
        return Reaction.neutral.value
    else:
        return Reaction.good.value


def predict_score(text: str) -> int:
    docs_new: list[str] = [text]
    X_new_counts = Config.count_vect.transform(docs_new)
    X_new_tfidf = Config.tfidf_transformer.transform(X_new_counts)
    y_pred: list[tuple[float, float]] = list(Config.clf.predict_proba(X_new_tfidf))
    return int(round(y_pred[0][1], 2) * 100)


use_checkpoint: bool = True
# generate_texts_for_prompts(use_checkpoint=use_checkpoint, checkpoint_dir=os.path.abspath("checkpoint-2100"))
# plot_word_cloud(use_checkpoint=use_checkpoint)
# evaluate_classifier()
