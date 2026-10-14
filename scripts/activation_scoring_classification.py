import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import transformers.models
import random
import numpy as np
from datasets import load_dataset
import logging
from tqdm import tqdm
import json
import argparse
from typing import Optional

# Setting args
parser = argparse.ArgumentParser()

parser.add_argument("--attnmask", type=str, default="causal", choices=["causal", "none", "row", "col"])
parser.add_argument("--dataid", type=str, default="obqa", choices=["arcc", "obqa", "boolq"])
parser.add_argument("--modelname", type=str, default="meta-llama/Llama-3.2-3B")
parser.add_argument("--inputpath", type=str, default="./out/obqa_processed")
parser.add_argument("--outputpath", type=str, default="./out")
parser.add_argument("--addbos", action='store_true')

args = parser.parse_args()

# Logging
logging.basicConfig(filename = f"./log_classification.txt",
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

attn_mask = args.attnmask
data_id = args.dataid
model_name = args.modelname
input_path = args.inputpath
output_path = args.outputpath
add_bos = args.addbos

logger.info(f"attn_mask: {attn_mask}, data_id: {data_id}, model_name: {model_name}, input_path: {input_path}, output_path: {output_path}, add_bos: {add_bos}")

# Setting seeds
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
set_seed(seed)

# Setting dataset
data_split = load_dataset(input_path)["test"]

# Setting model
device = torch.device("cuda")

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

model.config._attn_implementation="eager"
model.eval()

# Function from transformers v4.51.3: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/llama/modeling_llama.py, licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
# And https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py, licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Function adapted from transformers v4.51.3: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/llama/modeling_llama.py, licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
# And https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py, licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
def adapted_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    masking_start_index = 0,
    down=True,
    side=False,
    test_verbose=False,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        causal_mask[:, :, :, :] = 0
        
        if side:
            causal_mask[:, :, :, masking_start_index:] = torch.finfo(causal_mask.dtype).min

        attn_weights = attn_weights + causal_mask

    if down:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights[:, :, masking_start_index:] = 0
        attn_weights = attn_weights.to(query.dtype)
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    if test_verbose:
        logger.info("============= attn_weights =============")
        logger.info(attn_weights[0][0])
        logger.info(attn_weights.size())
        logger.info("==========================")
        logger.info("==========================")

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def find_sub_list(sl,l):
    """
    Function adapted from: https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list
    """
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return (ind,ind+sll-1)

@torch.no_grad()
def my_generate(model, tokenizer, input_str, test, attn_mask):
    input_to_input = tokenizer(input_str, return_tensors='pt')
    index_for_masking = find_sub_list(test, input_to_input["input_ids"][0].tolist())
    if index_for_masking is None:
        logger.info("ERROR TEST WASN'T FOUND!")

    input_to_model = input_to_input.to(device)
    if attn_mask == "none":
        out_model = model(**input_to_model, output_hidden_states=True, use_cache=False, masking_start_index=index_for_masking[0], down=False, side=False)["hidden_states"]
    elif attn_mask == "row":
        out_model = model(**input_to_model, output_hidden_states=True, use_cache=False, masking_start_index=index_for_masking[0], down=True, side=False)["hidden_states"]
    elif attn_mask == "col":
        out_model = model(**input_to_model, output_hidden_states=True, use_cache=False, masking_start_index=index_for_masking[0], down=False, side=True)["hidden_states"]
    else: # attn_mask == "casual"
        out_model = model(**input_to_model, output_hidden_states=True, use_cache=False)["hidden_states"]

    return out_model, index_for_masking[0]

def my_compute(input_t):
    abs_out = input_t.abs()
    
    per_v_max = torch.mean(torch.max(abs_out, dim=1).values).item()

    input_t_flatten = torch.flatten(input_t, start_dim=1)

    max_act = torch.mean(torch.max(input_t_flatten.abs(), dim=1).values).item()
    mean_abs = torch.mean(input_t_flatten.abs()).item()
    
    # Below adapted from: https://discuss.pytorch.org/t/statistics-for-whole-dataset/74511
    mean = torch.mean(input_t_flatten, dim=1, keepdim=True)
    diffs = input_t_flatten - mean
    var = torch.mean(torch.pow(diffs, 2.0), dim=1, keepdim=True)
    std = torch.sqrt(var)
    zscores = diffs / std
    skewness = torch.mean(torch.pow(zscores, 3.0), dim=(0, 1)).item()
    kurtosis = torch.mean(torch.pow(zscores, 4.0), dim=(0, 1)) - 3.0

    var_final = torch.mean(var).item()

    return [per_v_max,
            max_act, 
            mean_abs, 
            var_final,
            skewness,
            kurtosis.item()]

if attn_mask != "causal":
    if "Llama" in model_name:
        transformers.models.llama.modeling_llama.eager_attention_forward = adapted_eager_attention_forward
    elif "Qwen2" in model_name:
        transformers.models.qwen2.modeling_qwen2.eager_attention_forward = adapted_eager_attention_forward
    else:
        logger.info("============================== Error! Model used not supported. Using unadapted eager_attention_forward ==============================")

    ##############TEST (Verbose)##################
    input_to_input_test = tokenizer("Spain: Madrid\n\nUK:", return_tensors='pt')
    if add_bos:
        tokenized_test_test = tokenizer("UK:")["input_ids"][1:]
    else:
        tokenized_test_test = tokenizer("UK:")["input_ids"]
    index_for_masking_test = find_sub_list(tokenized_test_test, input_to_input_test["input_ids"][0].tolist())
    if index_for_masking_test is None:
        logger.info("ERROR TEST WASN'T FOUND!")
    logger.info(tokenized_test_test)
    logger.info(index_for_masking_test)
    with torch.no_grad():
        if attn_mask == "none":
            model(**input_to_input_test.to(device), output_hidden_states=True, use_cache=False, masking_start_index=index_for_masking_test[0], down=False, side=False, test_verbose=True)
        elif attn_mask == "row":
            model(**input_to_input_test.to(device), output_hidden_states=True, use_cache=False, masking_start_index=index_for_masking_test[0], down=True, side=False, test_verbose=True)
        elif attn_mask == "col":
            model(**input_to_input_test.to(device), output_hidden_states=True, use_cache=False, masking_start_index=index_for_masking_test[0], down=False, side=True, test_verbose=True)
        else:
            logger.info("============================== Error! That line should never be reached, did you manually change the code? If not, please ensure your attn_mask is one of the four supported maskings ==============================")
    ################################

progress_bar = tqdm(range(len(data_split)), desc="Generating")

per_v_max_all = {}
max_act_all = {}
mean_abs_all = {}
var_final_all = {}
skewness_all = {}
kurtosis_all = {}

for d in range(len(data_split)):
    # Prompt templates adapted from EleutherAI's LM Evaluation Harness (https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.8)
    if data_id == "obqa":
        formatted_prompt = data_split[d]["question_stem"]
    elif data_id == "arcc":
        formatted_prompt = data_split[d]["question"] + "\nAnswer:"
    elif data_id == "boolq":
        formatted_prompt = data_split[d]["question"] + "?\nAnswer:"
    else:
        logger.info("============================== Error! That line should never be reached, did you manually change the code? If not, please ensure your data_id is one of the three supported classification datasets ==============================")

    formatted_test = formatted_prompt.split("\n\n")[-1]
    
    current_id = data_split[d]["my_id"]

    if add_bos:
        tokenized_test = tokenizer(formatted_test)["input_ids"][1:]
    else:
        tokenized_test = tokenizer(formatted_test)["input_ids"]
        
    out_x, i_x = my_generate(model, tokenizer, formatted_prompt, tokenized_test, attn_mask)

    stacked = torch.stack([l.squeeze(0) for l in out_x])
    per_v_max, max_act, mean_abs, var_final, skewness, kurtosis =  my_compute(stacked)
    per_v_max_all[current_id] = per_v_max
    max_act_all [current_id] = max_act
    mean_abs_all[current_id] = mean_abs
    var_final_all[current_id] = var_final
    skewness_all[current_id] = skewness
    kurtosis_all[current_id] = kurtosis

    progress_bar.update(1)
    
with open(f'{output_path}/{attn_mask}_{data_id}.json', mode="w", encoding="utf-8") as write_file:
    json.dump([{"id": "per_v_max", "data": per_v_max_all}, {"id": "max_act", "data": max_act_all}, {"id": "mean_abs", "data": mean_abs_all}, {"id": "variance", "data": var_final_all}, {"id": "skewness", "data": skewness_all}, {"id": "kurtosis", "data": kurtosis_all}], write_file, indent=4)
