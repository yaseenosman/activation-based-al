# This code uses the BoolQ https://arxiv.org/abs/1905.00537, https://arxiv.org/abs/1905.10044, https://huggingface.co/datasets/aps/super_glue
# This code uses the prompt template used on EleutherAI's LM Evaluation Harness https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.8/lm_eval/tasks/super_glue/boolq/default.yaml, repository licensed under the MIT License (https://opensource.org/license/mit)
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import os

def format_shot(sample):
    """
    Given a sample, this function formats it based on the prompt template used in LM Evaluation Harness
    """
    possible_answers = ["no", "yes"]
    answer_text = possible_answers[sample['label']]
    return f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer: {answer_text}"

output_path = './out'

# Check if output path exists and if not create it
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loading data
print("Start processing aps/super_glue boolq!")
temp_train_split = load_dataset("aps/super_glue", 'boolq', split="train")
train_split = temp_train_split.shuffle(seed=42).select(list(range(1000)))

test_split = load_dataset("aps/super_glue", 'boolq', split="validation")

print("Loaded!")
print(f"Train size is: {len(train_split)}")
print(f"Test size is: {len(test_split)}")

# Processing data
progress_bar = tqdm(range(len(train_split)), desc="Processing dataset")

shot = format_shot(train_split[0])
updated_dataset = test_split.map(lambda example, idx: {"my_id": f"{0}.{idx}", "question": shot + "\n\n" + example["passage"] + "\nQuestion: " + example["question"]}, with_indices=True)
progress_bar.update(1)

for i in range(1, len(train_split)):
    shot = format_shot(train_split[i])
    temp_data = test_split.map(lambda example, idx: {"my_id": f"{i}.{idx}", "question": shot + "\n\n" + example["passage"] + "\nQuestion: " + example["question"]}, with_indices=True)
    updated_dataset = concatenate_datasets([updated_dataset, temp_data])
    progress_bar.update(1)

print(f"Length of processed data is: {len(updated_dataset)}")

updated_dataset.to_parquet(f"{output_path}/boolq_processed/test.parquet")
print("Completed!")
