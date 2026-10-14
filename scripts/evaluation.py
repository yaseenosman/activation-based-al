import logging
import lm_eval
import json
import argparse

# Setting args
parser = argparse.ArgumentParser()

parser.add_argument("--runid", type=str, default="obqa")
parser.add_argument("--modelname", type=str, default="meta-llama/Llama-3.2-3B")
parser.add_argument("--includepath", type=str, default="./lm_eval_tasks_adapted")
parser.add_argument("--task", type=str, default="obqa_sampling")
parser.add_argument("--outputpath", type=str, default="./out")
parser.add_argument("--addbos", action='store_true')
parser.add_argument("--batchsize", type=int, default=None)

args = parser.parse_args()

# Logging
logging.basicConfig(filename = "./log_evaluation.txt",
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

run_id = args.runid
model_name = args.modelname
include_path = args.includepath
task_name = args.task
output_path = args.outputpath
add_bos = args.addbos
batch_size = args.batchsize

# Running evaluation
logger.info(f"====================== Starting {run_id} evaluation ======================")
task_manager = lm_eval.tasks.TaskManager(include_path=include_path, file_path=None)
results = lm_eval.simple_evaluate(
    model="hf",
    model_args={"pretrained": model_name, "dtype": "auto", "add_bos_token": add_bos, "attn_implementation": "eager", "local_files_only": True},
    tasks=task_name,
    task_manager=task_manager,
    random_seed=42,
    numpy_random_seed=42,
    torch_random_seed=42,
    fewshot_random_seed=42,
    write_out=True,
    batch_size=batch_size,
)

# Post-processing results to reduce JSON file storage requirements
logger.info("Processing output...")
for j in range(len(results["samples"][task_name])):
    results["samples"][task_name][j]['doc'] = results["samples"][task_name][j]['doc']['my_id']
    results["samples"][task_name][j]['arguments'] = []

# Saving results
logger.info("Saving output...")
with open(f'{output_path}/{run_id}_evaluated.json', mode="w", encoding="utf-8") as write_file:
    json.dump(results["samples"], write_file)

logger.info(f"====================== Finished {run_id} evaluation ======================")
