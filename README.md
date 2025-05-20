# TRIAL: Token Relations and Importance Aware Late-interaction for Accurate Text Retrieval

# Preprocess dataset
## 0. Clean the corpus (remove documents with empty text)
```bash
python scripts/preprocess/clean_corpus.py
```
## 1. Split text to sentences

### Example: Split query in msmarco dataset to sentences
```bash
python scripts/preprocess/split_text_to_sentences.py +target_data=query dataset.name=beir-msmarco +op=split +total=1 +i=0
```
### Example: Split document in msmarco dataset to sentences
```bash
python scripts/preprocess/split_text_to_sentences.py +target_data=document dataset.name=beir-msmarco +op=split +total=1 +i=0
```


## 2. Tokenize sentences
Tokenization is done for both queries and documents. The tokenized data is saved in the same directory as the original data.

2.1. Set correct values for the variables in the script `scripts/bash/run_tokenization.sh`
```bash
begin=0
end=63
total=64
num_devices=2
```

2.2. Run the bash script to tokenize the sentences.
```bash
bash scripts/bash/run_tokenization.sh
```

2.3.
Merge the results
```bash
python scripts/preprocess/tokenize_data.py dataset.name=beir-msmarco +op=merge +total=${total} 
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
python scripts/preprocess/extract_phrases.py dataset.name=beir-msmarco +op=merge +total=${total} 
```

# Training
## Training the baseline model
```bash
python scripts/train.py _global.tag=colbert model=colbert training.use_torch_compile=True
```

## Training the EAGLE model
```bash
python scripts/train.py _global.tag=eagle model=eagle training.pad_to_max_length=False
```

# Indexing
```bash
python scripts/index.py _global.tag=colbert model=colbert model.ckpt_path=${CKPT_PATH} dataset.name=beir-msmarco
```

# Evaluation
## Evaluate with Reranking
```bash
python scripts/evaluate.py _global.tag=colbert model=colbert +args.ckpt_path=${CKPT_PATH} +args.mode=reranking
```

## Evaluate with Full Retrieval
```bash
python scripts/evaluate.py _global.tag=colbert model=colbert +args.ckpt_path=${CKPT_PATH} +args.mode=retrieval
```

# Inference
```bash
python scripts/inference.py _global.tag=colbert model=colbert +args.ckpt_path=${CKPT_PATH}
```

# Analysis
## Preliminary Study 1: Statistics of the number of words/phrases that are broken down into mulitple tokens
The number of broken word for queries and documents in the dataset.
```bash
python scripts/analysis/token_stats.py
```

The number of broken phrase for queries and documents in the dataset.
```bash
python scripts/analysis/phrase_stats.py +multiprocessing=True
```

## Preliminary Study 2: Statistics of the average scores for different types of POS tags
The average scores for different types of POS tags in the dataset.
```bash
python scripts/analysis/pos_stats.py +ckpt_path=/root/EAGLE/runs/colbert/best_model.ckpt
```

## Index Statistics
```bash
python scripts/analysis/index_stats.py
```

# Development
## Compare results of two different models
```bash
python scripts/develop/compare_two_results.py dataset.name=beir-arguana +args.ckpt1=${CKPT_PATH1} +args.ckpt2=${CKPT_PATH2}
```
