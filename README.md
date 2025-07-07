# FlopsRank
FLOPs-Normalized Efficiency-Effectiveness Metrics for LLM-based Rerankers

---
## Installation
Install via Conda (Python=3.11)
```bash
cd src
conda env create -f e2rflops.yml
```

The code base is adopted from [llm-reranker](https://github.com/ielab/llm-rankers). Refer to the page for more information on environment setup.

---

## Python code example:

```Python
from src import calculator

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
We provide the resulting files in [data](https://github.com/zhiyuanpeng/FlopsRank/tree/main/data).

---

Qwen2.5
3B, 7B, 14B



size_1(dl19)_numberOfchild(1(set=2), 2(set=3), 3(set=4))

#doc cut at 100

#
comparisons,input_length,docs_length,output_length,flops,time

comparisons:
for one query, the # of LLM-call to rerank 100 documents retrieved by BM25

input_length:
the number of input ids

doc_length:

a list of doc_length

output_length:
# 




