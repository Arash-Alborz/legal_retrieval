import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

lang = "nl"  # or "fr"
CORPUS_PATH = f"/content/drive/MyDrive/jina_embeddings/corpus_{lang}.csv"
OUTPUT_DIR = "/content/drive/MyDrive/jina_embeddings/embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_EMB = os.path.join(OUTPUT_DIR, f"jina_document_embeddings_{lang}.pt")
OUTPUT_IDS = os.path.join(OUTPUT_DIR, f"jina_document_ids_{lang}.json")

MODEL_NAME = "jinaai/jina-embeddings-v3"
BATCH_SIZE = 4  # GPU can handle higher, adjust as needed

df = pd.read_csv(CORPUS_PATH)
docs = df["article"].astype(str).tolist()
doc_ids = df["id"].astype(str).tolist()

print(f"Loaded {len(docs)} {lang.upper()} documents.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()

def embed_texts(texts, task="retrieval.passage", max_length=8192, truncate_dim=1024):
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            task=task,
            max_length=max_length,
            truncate_dim=truncate_dim,
        )
        embeddings = torch.from_numpy(embeddings)
    return embeddings.cpu()

all_embeddings = []
for i in tqdm(range(0, len(docs), BATCH_SIZE), desc=f"Embedding {lang.upper()} corpus"):
    batch_texts = docs[i:i+BATCH_SIZE]
    batch_emb = embed_texts(batch_texts, task="retrieval.passage")
    all_embeddings.append(batch_emb.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0)

torch.save(all_embeddings, OUTPUT_EMB)
with open(OUTPUT_IDS, "w", encoding="utf-8") as f:
    json.dump(doc_ids, f)