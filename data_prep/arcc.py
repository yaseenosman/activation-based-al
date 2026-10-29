# This code uses the ARC-Challenge https://arxiv.org/abs/1803.05457, https://huggingface.co/datasets/allenai/ai2_arc
# This code uses the prompt template used on EleutherAI's LM Evaluation Harness https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.8/lm_eval/tasks/arc/arc_easy.yaml, repository licensed under the MIT License (https://opensource.org/license/mit)
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import os

def format_shot(sample):
    """
    Given a sample, this function formats it based on the prompt template used in LM Evaluation Harness
    """
    answer_ind = sample['choices']['label'].index(sample['answerKey'])
    return f"Question: {sample['question']}\nAnswer: {sample['choices']['text'][answer_ind]}"

output_path = './out'

# Check if output path exists and if not create it
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loading data
print("Start processing allenai/ai2_arc!")
train_split = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split="train")
test_split = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split="test")

print("Loaded!")
print(f"Train size is: {len(train_split)}")
print(f"Test size is: {len(test_split)}")

# Processing data
progress_bar = tqdm(range(len(train_split)), desc="Processing dataset")

shot = format_shot(train_split[0])
updated_dataset = test_split.map(lambda example, idx: {"my_id": f"{0}.{idx}", "question": shot + "\n\nQuestion: " + example["question"]}, with_indices=True)
progress_bar.update(1)

for i in range(1, len(train_split)):
    shot = format_shot(train_split[i])
    temp_data = test_split.map(lambda example, idx: {"my_id": f"{i}.{idx}", "question": shot + "\n\nQuestion: " + example["question"]}, with_indices=True)
    updated_dataset = concatenate_datasets([updated_dataset, temp_data])
    progress_bar.update(1)

print(f"Length of processed data is: {len(updated_dataset)}")

updated_dataset.to_parquet(f"{output_path}/arcc_processed/test.parquet")
print("Completed!")
