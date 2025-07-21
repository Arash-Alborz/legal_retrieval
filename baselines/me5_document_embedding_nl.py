import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "/content/drive/MyDrive/me5_embeddings/corpus_fr.csv" 
OUTPUT_DIR = "/content/drive/MyDrive/me5_embeddings/embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_EMB = os.path.join(OUTPUT_DIR, "document_embeddings_fr.pt")
OUTPUT_IDS = os.path.join(OUTPUT_DIR, "document_ids_fr.json")

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
BATCH_SIZE = 64

df = pd.read_csv(CORPUS_PATH)
docs = df["article"].astype(str).tolist()
doc_ids = df["id"].astype(str).tolist()

print(f"Loaded {len(docs)} French documents from Google Drive.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

embeddings = []
for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding documents"):
    batch_texts = docs[i:i+BATCH_SIZE]
    batch_emb = model.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True)
    embeddings.append(batch_emb.cpu())

embeddings = torch.cat(embeddings, dim=0)

torch.save(embeddings, OUTPUT_EMB)
with open(OUTPUT_IDS, "w", encoding="utf-8") as f:
    json.dump(doc_ids, f)

print(f"Embeddings saved to: {OUTPUT_EMB}")
print(f"Document IDs saved to: {OUTPUT_IDS}")