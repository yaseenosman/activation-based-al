# Welcome to Activation-Based Active Learning for In-Context Learning
This repository will soon contain the official code for the paper: [Activation-Based Active Learning for In-Context Learning: Challenges and Insights](https://arxiv.org/pdf/2606.05134), accepted at Insights workshop, EMNLP 2026. Our webpage, which includes auxiliary results can be accessed through [here](https://yaseenosman.github.io/activation-based-al).


## Preparing Data
```bash
python ./data_prep/arcc.py
python ./data_prep/boolq.py
python ./data_prep/obqa.py
python ./data_prep/gsm8k.py
```

When we ran the experiments, we split each dataset into several segments to distribute across multiple GPUs in parallel. For GSM8K, as we used batching when calculating the shot scores, we provide the code to generate the 3 segments used to allow for reproducibility. For the remainder of datasets, because we did not use batching, we provide only the code for a single segment for simplicity. However, we split ARC-C, BoolQ, and OBQA data to 2, 4, and 2, respectively.

## Task Performance Evaluatation
The evaluation code can be found in `./scripts/evaluation.py` as well as under `./lm_eval_tasks_adapted`. Here we provide two example commands to run the evaluation of Llama-3.2-3B on GSM8K (first segment) and Qwen2.5-3B on OBQA.

```bash
python ./scripts/evaluation.py --runid "gsm8k" --modelname "meta-llama/Llama-3.2-3B" --includepath "./lm_eval_tasks_adapted" --task "gsm8k_sampling_1" --outputpath "./out" --addbos --batchsize 64 
```

```bash
python ./scripts/evaluation.py --runid "obqa" --modelname "Qwen/Qwen2.5-3B" --includepath "./lm_eval_tasks_adapted" --task "obqa_sampling" --outputpath "./out"
```