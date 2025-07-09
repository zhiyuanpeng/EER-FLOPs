# EER-FLOPs

Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers

---
## Installation
Install via Conda (Python=3.11)
```bash
conda env create -f e2rflops.yml
```

The code base is adopted from [llm-reranker](https://github.com/ielab/llm-rankers). Refer to the page for more information on environment setup.

---

## Data Description

The data folder `./data` contains experimental results for recording FLOP counts and latency on Qwen2.5 and Flan-T5 model series with the DL19 and DL20 datasets and various set sizes.

The files are named by [model_name]\_[dataset]\_[num_child].csv.

In each file, each row correpsonds to a query. The lists contain the results from running the setwise algorithm to rerank top-100 documents.

---

## Python code example:

```Python
from src.utils import calculator

# Qwen2.5-7B configuration
input_length = 200
output_length = 1
num_layers = 28
model_dim = 3584
ffsize = 18944
ratio = 7

flopscount = calculator.flops_decoder(input_length, output_length, num_layers, model_dim, ffsize, ratio)
print(flopscount)
```
---

## Experiment examples (TREC DL19-20)
### Step 1: BM25 Retrieval via [Pyserini](https://github.com/castorini/pyserini)

Example on TREC DL 2019:
```bash
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl19-passage \
  --output run.msmarco-v1-passage.bm25-default.dl19.txt \
  --bm25 --k1 0.9 --b 0.4
```

---

### Step 2: Record FLOP Counts for Top-100 Reranking

Example for Flan-T5-Large on DL19 with a set size of 2:
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/run.py \
  run --model_name_or_path google/flan-t5-large \
      --tokenizer_name_or_path google/flan-t5-large \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.setwise.heapsort.k10.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 100 \
      --scoring generation \
      --device cuda \
  setwise --num_child 1 \
          --method heapsort \
          --k 10
```
We provide the resulting files in [data](./data).

