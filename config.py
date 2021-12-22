import os.path

import torch
from transformers import PreTrainedTokenizerFast, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer


class Config:
    _model: PreTrainedModel = None
    _tokenizer: PreTrainedTokenizerFast = None
    data_dir: str = os.path.abspath("data")
    bibtex_dir: str = os.path.join(data_dir, "bibTex")
    dataset_path: str = os.path.join(data_dir, "dataset_tok.txt")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_generation_length: int = 40
    max_sequence_length: int = 1024
    model_name: str = "benjamin/gerpt2"  # -large "deepset/gelectra-large" gbert
    models_dir: str = os.path.abspath("models")
    output_dir: str = os.path.abspath("checkpoints")
    outputs_path: str = os.path.abspath("outputs.json")
    pdf_dir: str = os.path.join(data_dir, "pdf")
    plots_dir: str = os.path.abspath("plots")
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
