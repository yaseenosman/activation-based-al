import json
import numpy as np
from scipy.stats import spearmanr
import argparse
import os

# Setting args
parser = argparse.ArgumentParser()

parser.add_argument("--attnmask", type=str, default="causal", choices=["causal", "none", "row", "col"])
parser.add_argument("--dataid", type=str, default="obqa", choices=["arcc", "obqa", "boolq"])
parser.add_argument("--metricname", type=str, default="acc") # exact_match for gsm8k
parser.add_argument("--inputpath", type=str, default="./out")
parser.add_argument("--outputpath", type=str, default="./out")

args = parser.parse_args()

attn_mask = args.attnmask
data_id = args.dataid
metric = args.metricname
input_path = args.inputpath
output_path = args.outputpath

def get_all_data_dir_classification(input_path, data_id, attn_mask, metric):
    shots = {}
    print(f"Loading {data_id}...")
    with open(f'{input_path}/{data_id}_evaluated.json', mode="r", encoding="utf-8") as f:
        evaluations_data = json.load(f)
    evaluations_key = list(evaluations_data.keys())[0]
    with open(f'{input_path}/{attn_mask}_{data_id}.json', mode="r", encoding="utf-8") as f:
        scores_data = json.load(f)

    for i in range(len(evaluations_data[evaluations_key])):
        k = evaluations_data[evaluations_key][i]['doc']
        shot_num = k.split(".")[0]

        if shot_num in shots:
            for index_score in range(6):
                shots[shot_num][index_score].append(scores_data[index_score]["data"][k])
            shots[shot_num][6].append(evaluations_data[evaluations_key][i][metric])
        else:
            shots[shot_num] = [[scores_data[0]["data"][k]], [scores_data[1]["data"][k]], [scores_data[2]["data"][k]], 
                               [scores_data[3]["data"][k]], [scores_data[4]["data"][k]], [scores_data[5]["data"][k]], [evaluations_data[evaluations_key][i][metric]]]
    
    print(f"Number of shots: {len(shots)}")
    act_score = []
    for s in shots:
        act_score.append([np.array(shots[s][0]).mean().item(), np.array(shots[s][1]).mean().item(), np.array(shots[s][2]).mean().item(),
                          np.array(shots[s][4]).mean().item(), np.array(shots[s][4]).mean().item(), np.array(shots[s][5]).mean().item(), np.array(shots[s][6]).mean().item()])

    act_score = np.array(act_score)
    print(f"act_score shape: {act_score.shape}")
    return act_score

def get_all_data_dir_generative(input_path, data_id, attn_mask, metric):
    file_index = 1
    shots = {}
    while f"{data_id}{file_index}_evaluated.json" in os.listdir(input_path):
        print(f"Loading {data_id}{file_index}...")
        with open(f'{input_path}/{data_id}{file_index}_evaluated.json', mode="r", encoding="utf-8") as f:
            evaluations_data = json.load(f)
        evaluations_key = list(evaluations_data.keys())[0]
        with open(f'{input_path}/{attn_mask}_{data_id}{file_index}.json', mode="r", encoding="utf-8") as f:
            scores_data = json.load(f)

        for i in range(len(evaluations_data[evaluations_key])):
            # Used strict-match as the filter choice following: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions/670
            if evaluations_data[evaluations_key][i]["filter"] == "strict-match":
                k = evaluations_data[evaluations_key][i]['doc']
                shot_num = k.split(".")[0]

                if shot_num in shots:
                    for index_score in range(6):
                        shots[shot_num][index_score].append(scores_data[index_score]["data"][k])
                    shots[shot_num][6].append(evaluations_data[evaluations_key][i][metric])
                else:
                    shots[shot_num] = [[scores_data[0]["data"][k]], [scores_data[1]["data"][k]], [scores_data[2]["data"][k]], 
                                       [scores_data[3]["data"][k]], [scores_data[4]["data"][k]], [scores_data[5]["data"][k]], [evaluations_data[evaluations_key][i][metric]]]
        file_index += 1
    
    print(f"Number of shots: {len(shots)}")
    act_score = []
    for s in shots:
        act_score.append([np.array(shots[s][0]).mean().item(), np.array(shots[s][1]).mean().item(), np.array(shots[s][2]).mean().item(),
                          np.array(shots[s][4]).mean().item(), np.array(shots[s][4]).mean().item(), np.array(shots[s][5]).mean().item(), np.array(shots[s][6]).mean().item()])

    act_score = np.array(act_score)
    print(f"act_score shape: {act_score.shape}")
    return act_score

def get_person_corr(act_score):
    correlations = []
    for c in range(6):
        corr_value, p_value = spearmanr(act_score[:, c], act_score[:, 6])
        correlations.append([corr_value, p_value])
    correlations = np.array(correlations)
    print(f"correlations shape: {correlations.shape}")
    return correlations

if data_id == "gsm8k":
    output_correlations = get_person_corr(get_all_data_dir_generative(input_path, data_id, attn_mask, metric))
else:
    output_correlations = get_person_corr(get_all_data_dir_classification(input_path, data_id, attn_mask, metric))

with open(f'{output_path}/{attn_mask}_{data_id}_correlations.json', mode="w", encoding="utf-8") as write_file:
    json.dump([{"correlations": output_correlations}], write_file, indent=4)
