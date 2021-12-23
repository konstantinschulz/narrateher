import os.path

import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from transformers import PreTrainedTokenizerFast, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer


class Config:
    data_dir: str = os.path.abspath("data")
    models_dir: str = os.path.abspath("models")
    _model: PreTrainedModel = None
    _tokenizer: PreTrainedTokenizerFast = None
    anti_feminism_dir: str = os.path.join(data_dir, "anti-feminism")
    bibtex_dir: str = os.path.join(data_dir, "bibTex")
    clf: MultinomialNB = MultinomialNB()
    count_vect: CountVectorizer = CountVectorizer()
    dataset_path: str = os.path.join(data_dir, "dataset_tok.txt")
    # device: torch.device = torch.device("cpu")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feminism_detection_dataset_path: str = os.path.join(data_dir, "feminism_detection_dataset.jsonl")
    max_generation_length: int = 40
    max_sequence_length: int = 1024
    model_checkpoint_path: str = os.path.join(models_dir, "model.pt")
    model_name: str = "benjamin/gerpt2-large"  # -large "deepset/gelectra-large" gbert
    output_dir: str = os.path.abspath(".")
    outputs_path: str = os.path.abspath("outputs.json")
    pdf_dir: str = os.path.join(data_dir, "pdf")
    plots_dir: str = os.path.abspath("plots")
    tfidf_transformer: TfidfTransformer = TfidfTransformer()
    tok_dir: str = os.path.join(data_dir, "tok")
    txt_dir: str = os.path.join(data_dir, "txt")

    @classmethod
    def model(cls) -> PreTrainedModel:
        if not Config._model:
            Config._model = AutoModelForCausalLM.from_pretrained(Config.model_name).to(Config.device)
        return Config._model

    @classmethod
    def tokenizer(cls) -> PreTrainedTokenizerFast:
        if not Config._tokenizer:
            Config._tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        return Config._tokenizer


os.makedirs(Config.bibtex_dir, exist_ok=True)
os.makedirs(Config.pdf_dir, exist_ok=True)
os.makedirs(Config.plots_dir, exist_ok=True)
os.makedirs(Config.tok_dir, exist_ok=True)
os.makedirs(Config.txt_dir, exist_ok=True)
