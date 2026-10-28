# Welcome to Activation-Based Active Learning for In-Context Learning
This repository contains the official code for the paper: [Activation-Based Active Learning for In-Context Learning: Challenges and Insights](https://arxiv.org/pdf/2606.05134), accepted at Insights workshop, EMNLP 2026. Our webpage, which includes auxiliary results can be accessed through [here](https://yaseenosman.github.io/activation-based-al).

## Setup
Please follow the steps below to setup the environment and code.

### Clone code
```bash
git clone https://github.com/yaseenosman/activation-based-al.git
cd activation-based-al
```

### Setup Environment
Create an activate an envrionment with Python 3.12.6:
```bash
conda create -n activation-al python==3.12.6
conda install -n activation-al pip
conda activate activation-al
```

Then install the libraries:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install triton==3.2.0
pip install transformers==4.51.3
pip install matplotlib
pip install lm_eval==0.4.8
```
We have used CUDA 12.4.0

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
The `--runid` value can be one of the following: `["arcc_sampling", "boolq_sampling", "obqa_sampling", "gsm8k_sampling_1", "gsm8k_sampling_2", "gsm8k_sampling_3"]`, representing the task/segment to be run. We use `--addbos` when running Llama-3.2-3B, but not when running Qwen2.5-3B.

## Computing Shot Scores
The code to compute the shot scores can be found in `./scripts/activation_scoring_classification.py` and `./scripts/activation_scoring_generative.py`. Below are example commands to run

```bash
python ./scripts/activation_scoring_generative.py --attnmask "none" --dataid "gsm8k" --splitid 1 --modelname "meta-llama/Llama-3.2-3B" --inputpath "./out/gsm8k_processed_1" --outputpath "./out" --addbos
```

```bash
python ./scripts/activation_scoring_classification.py --attnmask "causal" --dataid "obqa" --modelname "Qwen/Qwen2.5-3B" --inputpath "./out/obqa_processed" --outputpath "./out"
```

The following `--attnmask` options are available as options: `["causal", "none", "row", "col"]`. For `./scripts/activation_scoring_classification.py`, available `--dataid` options are: `["arcc", "obqa", "boolq"]`. Conversely, `./scripts/activation_scoring_generative.py` has only `"gsm8k"` as an option. Again, we use `--addbos` when running Llama-3.2-3B, but not when running Qwen2.5-3B.

## Calculating Spearman Correlation
<!-- Place holder-->

## Acknowledgement
This work was supported by the UK Research and Innovation Centre for Doctoral Training in Machine Intelligence for Nano-electronic Devices and Systems [EP/S024298/1]. The authors acknowledge the use of the IRIDIS High Performance Computing Facility, and associated support services at the University of Southampton, in the completion of this work.

This repository benefitted from [transformers by huggingface](https://github.com/huggingface/transformers) and [lm-evaluation-harness by EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness) codes.

We acknowledge the help of Dr Corran Webster from the Southampton Research Software Group and the MINDS CDT Impact team on reviewing the code.

## Citation
If you find our work useful, please consider citing our paper.
```bibtex
@article{osman2026activation,
  title     = {Activation-Based Active Learning for In-Context Learning: Challenges and Insights},
  author    = {Osman, Yaseen M and Merrett, Geoff V and Middleton, Stuart E},
  journal   = {arXiv preprint arXiv:2606.05134},
  year      = {2026},
}
```