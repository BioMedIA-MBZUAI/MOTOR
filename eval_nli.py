import json
import os
import argparse
import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SEED = 0
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(SEED)
torch.random.manual_seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output_file",
    type=str,
    help="Outpur file path (.csv)",
    required=True,
)

args = parser.parse_args()

nli_model_path = "razent/SciFive-large-Pubmed_PMC-MedNLI"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
nli_model = AutoModelForSeq2SeqLM.from_pretrained(nli_model_path)
nli_model.to("cuda")
nli_model.eval()

output_file_name = args.output_file.split("/")[-1]
if "no_rag" in output_file_name or "exact" in output_file_name:
    parent_folder = args.output_file.split("/")[-2]
    outupts_folder = args.output_file.split("/")[-3]
else:
    parent_folder = (
        args.output_file.split("/")[-3] + "/" + args.output_file.split("/")[-2]
    )
    outupts_folder = args.output_file.split("/")[-4]


def compute_score(question, ground_truth, generated_answer):
    try:
        ground_truth = question + " Answer is: " + ground_truth
        generated_answer = question + " Answer is: " + generated_answer
        text = f"mednli: sentence1: {ground_truth} sentence2: {generated_answer}"
        encoding = nli_tokenizer.encode_plus(
            text, padding="max_length", max_length=256, return_tensors="pt"
        )
        input_ids, attention_masks = encoding["input_ids"].to(("cuda")), encoding[
            "attention_mask"
        ].to("cuda")

        outputs = nli_model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=8,
            early_stopping=True,
        )

        for output in outputs:
            line = nli_tokenizer.decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            if "contradiction" in line.lower():
                return 0.0  # Contradictory
            else:
                return 1.0  # Entailment or Neutral
    except Exception as e:
        print(e)
        return 0.0


# Load the data
data = pd.read_csv(f"{args.output_file}.csv")

# Evaluate each row in the data
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    question = row["Question"]
    if "Retrieved Context" in data.columns:
        context = row["Retrieved Context"]
    else:
        context = None
    ground_truth = row["Answer"]
    if "RAG Response" in data.columns:
        generated_answer = row["RAG Response"]
    else:
        generated_answer = row["Response"]

    evaluation_score = compute_score(question, ground_truth, generated_answer)

    print("\nCorrectness Score: ", evaluation_score)
    data.at[index, "Correctness Score"] = evaluation_score
    data.at[index, "Best Score"] = 1


# Calculate the final scores
y_true = data["Best Score"].astype(int)
correctness_y_pred = data["Correctness Score"].astype(int)
correctness_accuracy = accuracy_score(y_true, correctness_y_pred)

scores = {
    "Correctness Accuracy": correctness_accuracy,
}

print("\nFinal Scores: ", scores)

# Save scores to a JSON file
scores_json = json.dumps(scores)
with open(
    f"{outupts_folder}/{parent_folder}/nli_{output_file_name}_scores.json", "w"
) as file:
    file.write(scores_json)

# Save scores to a CSV file
data.to_csv(
    f"{outupts_folder}/{parent_folder}/nli_{output_file_name}_scores.csv",
    index=False,
)

print("Successfully evaluated: ", output_file_name)
