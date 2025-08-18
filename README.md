# LLM Legal Document Retrieval

This repository contains the code, evaluation scripts, and results for the thesis project on **Legal Document Retrieval using Large Language Models (LLMs)**.  
The project evaluates multiple retrieval and ranking strategies across different models (Gemini, GPT, LLaMA, Qwen, Jina, mE5, etc.) on Belgian legal texts.

---

## 📂 Project Structure

llm_legal_document_retrieval/
│
├── baselines/                  # baseline models and results (BM25, TF-IDF, etc.)
│
├── data_processing/             # preprocessing scripts and cleaned datasets
│
├── hard_negatives_stats/        # statistics and analysis of hard negatives
│
├── llm_retrieval/               # main retrieval scripts with LLMs
│
├── ranking/                     # reranking with embeddings (Jina, mE5, etc.)
│
├── results/                     # final evaluation results and CSV summaries
│   └── plots/                   # evaluation plots and figures
│
├── retrievals/                  # raw retrieval outputs (per model & scenario)
│
├── sampling_hard_negatives/     # generated hard negatives for evaluation
│
├── .gitignore                   # ignored files (large embeddings, datasets, etc.)
├── environment.yml              # conda environment definition
├── extra_codes.ipynb            # additional experimental notebooks
└── requirements.txt             # Python dependencies

---

## ⚙️ Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/Arash-Alborz/legal_retrieval.git
cd legal_retrieval

# Option 1: create environment via conda
conda env create -f environment.yml
conda activate llm_legal_document_retrieval

# Option 2: install requirements directly
pip install -r requirements.txt

🚀 Usage

Retrieval & Ranking

Scripts for different scenarios are in /scripts/:
	•	001_example_id_retrieval_v01.py – ID-based retrieval
	•	002_example_rc_retrieval_v01.py – Relevance classification retrieval
	•	003_example_pp_ranking_v01.py – Pointwise ranking
	•	004_example_lw_ranking_v01.py – Listwise ranking
	•	005_prompt_builder_all_scenarios_v01.py – unified prompt builder
	•	006_jina_reranking_v01.py – reranking with Jina embeddings
	•	007_me5_reranking_v01.py – reranking with mE5 embeddings

Evaluation

Evaluation scripts are in /eval/:
	•	001_retrieval_evaluation_v01.py – evaluates retrieval (Precision, Recall, F1)
	•	002_ranking_evaluation_v01.py – evaluates ranking (R@k, MAP, MRR, nDCG)

🔗 Reference

GitHub: https://github.com/Arash-Alborz/legal_retrieval