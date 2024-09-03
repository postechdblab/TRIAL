# EAGLE: Efficient Retrieval Using Aggregated Scores from Multiple Granularity Embeddings

# Preprocess dataset
## 1. Split text to sentences

### Example: Split query in msmarco dataset to sentences
```bash
python scripts/preprocess/split_text_to_sentences.py +target_data=query dataset.name=beir-msmarco
```
### Example: Split document in msmarco dataset to sentences
```bash
python scripts/preprocess/split_text_to_sentences.py +target_data=document dataset.name=beir-msmarco
```

## 2. Extract Phrases

### Example: Extract phrase indices from query in msmarco dataset
```bash
python scripts/preprocess/extract_phrases.py \
+target_data=query \
+total=1 \
+i=0 \
+op=extract \
dataset.name=beir-msmarco \
```