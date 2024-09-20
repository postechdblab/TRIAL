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


## 2. Tokenize sentences
Tokenization is done for both queries and documents. The tokenized data is saved in the same directory as the original data.
```bash
python scripts/preprocess/tokenize_data.py \
+total=1 \
+i=0 \
+op=tokenize \
dataset.name=beir-msmarco
```


## 3. Extract Phrases

### Extract phrase range

3.1. Set correct values for the variables in the script
```bash
begin=0
end=31
total=166
num_devices=8
```

3.2. Run the bash script to extract phrases. 

This will start multiple processes and create multiple files containing the phrase ranges
```bash
bash scripts/bash/run_phrase_extraction.sh
```

### Merge the splitted files into one
```bash
python scripts/preprocess/extract_phrase.py +op=merge +total=${total} 
```
