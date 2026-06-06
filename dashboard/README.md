# Sentiment Analysis Dashboard — COS60011

Interactive dashboard for the project demo (Microsoft Teams live presentation).


## Quick Start to run the dashboard locally.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate processed data splits (run from project root)
python src/data_preprocessing.py

# 3. Run the dashboard
streamlit run app.py

```

The dashboard opens at `https://technologydesignproject.streamlit.app/`.


```

## Tab Ownership

| Tab | Owner | Results CSV |
|-----|-------|-------------|
| Overview | Fin | — |
| Data Pipeline | Bikram | live processed CSVs |
| Baselines | Hanok | `task2_baseline_results.csv` |
| BERT Fine-tuning | Aniketh | `bert_results.csv` |
| Sequential Transfer | Fin / Bikram | `task4_transfer_results.csv` |
| LLM Experiments | Kevin | `llm_results.csv` |
| Evaluation | Himanshu | all CSVs above |

