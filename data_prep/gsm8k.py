# This code uses the BoolQ https://arxiv.org/abs/2110.14168, https://huggingface.co/datasets/openai/gsm8k
# This code uses the prompt template used on EleutherAI's LM Evaluation Harness https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.8/lm_eval/tasks/gsm8k/gsm8k.yaml, repository licensed under the MIT License (https://opensource.org/license/mit)
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import os

def format_shot(sample):
    """
    Given a sample, this function formats it based on the prompt template used in LM Evaluation Harness
    """
    return f"Question: {sample['question']}\nAnswer: {sample['answer']}"

output_path = './out'

# Check if output path exists and if not create it
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loading data
print("Start processing openai/gsm8k!")
temp_train_split = load_dataset("openai/gsm8k", "main", split="train")
train_split = temp_train_split.shuffle(seed=42).select(list(range(1000)))

test_split = load_dataset("openai/gsm8k", "main", split="test")

print("Loaded!")
print(f"Train size is: {len(train_split)}")
print(f"Test size is: {len(test_split)}")

# Processing data
progress_bar = tqdm(range(len(train_split)), desc="Processing dataset")

def format_seqment(train_split, start_index, end_index, progress_bar):
    shot = format_shot(train_split[start_index])
    updated_dataset = test_split.map(lambda example, idx: {"my_id": f"{start_index}.{idx}", "question": shot + "\n\nQuestion: " + example["question"]}, with_indices=True)
    progress_bar.update(1)
    for i in range(start_index+1, end_index):
        shot = format_shot(train_split[i])
        temp_data = test_split.map(lambda example, idx: {"my_id": f"{i}.{idx}", "question": shot + "\n\nQuestion: " + example["question"]}, with_indices=True)
        updated_dataset = concatenate_datasets([updated_dataset, temp_data])
        progress_bar.update(1)
    return updated_dataset

# Processing first part
updated_dataset = format_seqment(train_split, 0, len(train_split)//3, progress_bar)

print(f"Length of first part of processed data is: {len(updated_dataset)}")

updated_dataset.to_parquet(f"{output_path}/gsm8k_processed_1/test.parquet")
print("Completed Part 1!")

# Processing second part
updated_dataset = format_seqment(train_split, len(train_split)//3, len(train_split)*2//3, progress_bar)

print(f"Length of second part of processed data is: {len(updated_dataset)}")

updated_dataset.to_parquet(f"{output_path}/gsm8k_processed_2/test.parquet")
print("Completed Part 2!")

# Processing third part
updated_dataset = format_seqment(train_split, len(train_split)*2//3, len(train_split), progress_bar)

print(f"Length of third part of processed data is: {len(updated_dataset)}")

updated_dataset.to_parquet(f"{output_path}/gsm8k_processed_3/test.parquet")
print("Completed Part 3!")
