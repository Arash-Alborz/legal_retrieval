import os
import pandas as pd
from tqdm import tqdm
import spacy

tqdm.pandas(desc="Preprocessing")

class TextPreprocessor:
    def __init__(self, spacy_model: str):
        self.nlp = spacy.load(spacy_model)

    def preprocess_series(self, series: pd.Series) -> pd.Series:
        return series.progress_apply(self.preprocess_text)

    def preprocess_text(self, text: str) -> str:
        doc = self.nlp(text.lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and not token.like_num
        ]
        return " ".join(tokens)

def preprocess_and_save(lang: str, corpus_path: str, queries_path: str, output_dir: str, spacy_model: str):
    print(f"\nProcessing language: {lang.upper()}")

    preprocessor = TextPreprocessor(spacy_model)

    # Load data
    corpus_df = pd.read_csv(corpus_path)
    queries_df = pd.read_csv(queries_path)

    # Preprocess
    print("Preprocessing articles...")
    corpus_df["article"] = preprocessor.preprocess_series(corpus_df["article"])
    print("Preprocessing queries...")
    queries_df["question"] = preprocessor.preprocess_series(queries_df["question"])

    # Save
    os.makedirs(output_dir, exist_ok=True)

    corpus_out = os.path.join(output_dir, f"corpus_{lang}_clean.csv")
    queries_out = os.path.join(output_dir, f"queries_{lang}_clean.csv")

    corpus_df.to_csv(corpus_out, index=False, encoding="utf-8")
    queries_df.to_csv(queries_out, index=False, encoding="utf-8")

    print(f"Saved preprocessed corpus to: {corpus_out}")
    print(f"Saved preprocessed queries to: {queries_out}")

if __name__ == "__main__":
    output_dir = "preprocessed_data"

    paths = {
    "fr": {
        "corpus": "../data_processing/data/original_csv/corpus_fr.csv",
        "queries": "../data_processing/data/cleaned_queries_csv/cleaned_test_queries_fr.csv",
        "spacy_model": "fr_core_news_md",
    },
    "nl": {
        "corpus": "../data_processing/data/original_csv/corpus_nl.csv",
        "queries": "../data_processing/data/cleaned_queries_csv/cleaned_test_queries_nl.csv",
        "spacy_model": "nl_core_news_md",
    }
}

    for lang in ["fr", "nl"]:
        preprocess_and_save(
            lang=lang,
            corpus_path=paths[lang]["corpus"],
            queries_path=paths[lang]["queries"],
            output_dir=output_dir,
            spacy_model=paths[lang]["spacy_model"]
        )